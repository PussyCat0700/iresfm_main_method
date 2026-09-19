#!/usr/bin/env python3
"""Run and evaluate IresFM on the public 105-site data."""

from __future__ import annotations

import argparse
import functools
import gc
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from onnx import numpy_helper
from safetensors.numpy import load_file as load_safetensors


TZ_SHIFT = pd.Timedelta(hours=8)
POWER_STEP = pd.Timedelta(minutes=15)
POWER_HISTORY = 480
WEATHER_HISTORY = 120
WEATHER_FUTURE = 120
FORECAST_STEPS = 480
TP_INDEX = 20
SSR_INDEX = 26
RAW_CUMULATIVE_SSR_THRESHOLD = 1.0e7
SITE_METRIC_COLUMNS = ("MAE_percent", "RMSE_percent", "R2", "Corr")
PACKAGE_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=["inspect", "forecast", "evaluate", "summarize", "plot"],
    )
    parser.add_argument(
        "--data-root", type=Path, default=PACKAGE_ROOT / "data/iresfm_open_105"
    )
    parser.add_argument(
        "--model", type=Path, default=PACKAGE_ROOT / "models/iresfm_base.onnx"
    )
    parser.add_argument(
        "--model-state",
        choices=["zero-shot", "fine-tuned"],
        default="zero-shot",
    )
    parser.add_argument("--heads-dir", type=Path, default=PACKAGE_ROOT / "heads")
    parser.add_argument("--station-id", type=int, action="append")
    parser.add_argument("--all-sites", action="store_true")
    parser.add_argument(
        "--issue-time",
        help="Local issue time; omitted: select a time within the site's test period",
    )
    parser.add_argument(
        "--task",
        choices=["full", "ultra_short", "short", "medium"],
        default="short",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/forecast.csv"))
    parser.add_argument("--zero-shot-forecast", type=Path)
    parser.add_argument("--fine-tuned-forecast", type=Path)
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path("outputs/forecast_comparison.png"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluation"),
    )
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def time_encoding(
    length: int,
    dimensions: int,
    start: float,
    step: float,
) -> np.ndarray:
    positions = start + np.arange(length, dtype=np.float32) * step
    indices = np.arange(dimensions, dtype=np.float32)
    scales = np.power(10000.0, indices / float(dimensions))
    angles = positions[:, None] / scales[None, :]
    encoding = np.empty((length, dimensions), dtype=np.float32)
    encoding[:, 0::2] = np.sin(angles[:, 0::2])
    encoding[:, 1::2] = np.cos(angles[:, 1::2])
    return encoding


def repair_cumulative_weather(weather: np.ndarray) -> np.ndarray:
    if float(np.nanmax(weather[..., SSR_INDEX])) <= RAW_CUMULATIVE_SSR_THRESHOLD:
        return weather
    repaired = np.asarray(weather, dtype=np.float32).copy()
    for index in (TP_INDEX, SSR_INDEX):
        repaired[1:, :, index] -= repaired[:-1, :, index]
    return repaired


def available_forecast_base(
    issue_time_bjt: pd.Timestamp,
) -> tuple[pd.Timestamp, int]:
    issue_utc = (issue_time_bjt - TZ_SHIFT).floor("h")
    day = issue_utc.floor("D")
    base_06 = day + pd.Timedelta(hours=6)
    base_18 = day + pd.Timedelta(hours=18)
    if issue_utc >= base_18:
        base = base_18
    elif issue_utc >= base_06:
        base = base_06
    else:
        base = day - pd.Timedelta(days=1) + pd.Timedelta(hours=18)
    offset = int((issue_utc - base).total_seconds() // 3600)
    return base, offset


class PublicSite:
    def __init__(self, data_root: Path, station_id: int):
        self.data_root = data_root
        self.station_id = int(station_id)
        info = pd.read_csv(data_root / "info.csv")
        match = info.loc[info["station_id"].astype(int) == self.station_id]
        if len(match) != 1:
            raise ValueError(f"station_id {station_id} is not unique")
        self.info = match.iloc[0]
        self.label = int(self.info["type_id"])

        power = pd.read_csv(
            data_root / "active_power_norm" / f"{station_id}.csv"
        )
        power["time"] = pd.to_datetime(power["time"], errors="raise")
        power["power"] = pd.to_numeric(power["power"], errors="coerce")
        self.power = (
            power.sort_values("time")
            .drop_duplicates("time", keep="last")
            .set_index("time")["power"]
            .astype(np.float32)
        )
        complete_index = pd.date_range(
            self.power.index.min(),
            self.power.index.max(),
            freq="15min",
        )
        self.power = self.power.reindex(complete_index)

        self.train_start = pd.Timestamp(self.info["train_start"])
        self.train_end = pd.Timestamp(self.info["train_end"])
        self.test_start = pd.Timestamp(self.info["test_start"])
        self.test_end = pd.Timestamp(self.info["test_end"])
        training_power = self.power.loc[self.train_start : self.train_end].dropna()
        self.power_mean = float(training_power.mean())
        self.power_std = float(training_power.std(ddof=1))
        if not np.isfinite(self.power_std) or self.power_std <= 0:
            raise ValueError(f"station {station_id}: invalid power scale")

        normalization = np.load(
            data_root / "model_aux" / "weather_normalization.npz"
        )
        matches = np.flatnonzero(
            normalization["station_id"].astype(np.int64) == self.station_id
        )
        if len(matches) != 1:
            raise ValueError(f"station {station_id}: weather statistics not found")
        row = int(matches[0])
        self.weather_mean = normalization["mean"][row].astype(np.float32)
        self.weather_std = np.maximum(
            normalization["std"][row].astype(np.float32),
            1.0e-6,
        )
        self.relative_bias = np.load(
            data_root / "model_aux" / "relative_bias.npy",
            mmap_mode="r",
        )[row].astype(np.float32)
        self.weather_mask = np.load(
            data_root / "model_aux" / "weather_var_mask.npy",
            mmap_mode="r",
        )[row].astype(np.float32)

        self.weather_past_encoding = time_encoding(
            WEATHER_HISTORY, 27, 0, 1
        )
        self.weather_future_encoding = time_encoding(
            WEATHER_FUTURE, 27, WEATHER_HISTORY + 1, 1
        )
        self.power_encoding = time_encoding(POWER_HISTORY, 2, 0, 0.25)
        self._reference_weather: np.ndarray | None = None
        self._reference_weather_start: pd.Timestamp | None = None

    def load_reference_weather(self) -> None:
        if self._reference_weather is not None:
            return
        directory = self.data_root / "weather_finetune" / str(self.station_id)
        files = sorted(directory.glob("*.npy"))
        if not files:
            raise FileNotFoundError(f"no reference weather under {directory}")
        weather = np.concatenate(
            [
                np.load(path, allow_pickle=False).astype(
                    np.float32, copy=False
                )
                for path in files
            ],
            axis=0,
        )
        first = files[0].name.rsplit(f"_{self.station_id}.npy", 1)[0]
        self._reference_weather_start = pd.Timestamp(
            first.replace("_", " ", 1)
        )
        weather = (
            weather - self.weather_mean[None, None, :]
        ) / self.weather_std[None, None, :]
        weather *= self.weather_mask[None, None, :]
        self._reference_weather = np.nan_to_num(
            weather,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)

    def reference_weather_at(self, time_bjt: pd.Timestamp) -> np.ndarray:
        self.load_reference_weather()
        assert self._reference_weather is not None
        assert self._reference_weather_start is not None
        time_utc = (time_bjt - TZ_SHIFT).floor("h")
        position = int(
            (time_utc - self._reference_weather_start).total_seconds() // 3600
        )
        if not 0 <= position < len(self._reference_weather):
            raise ValueError(
                f"station {self.station_id}: no weather for {time_bjt}"
            )
        return self._reference_weather[position]

    def power_input(self, issue_time: pd.Timestamp) -> np.ndarray:
        positions = self.power.index.get_indexer([issue_time])
        if positions[0] < POWER_HISTORY - 1:
            raise ValueError(
                f"station {self.station_id}: insufficient power history"
            )
        end = int(positions[0])
        values = self.power.to_numpy(np.float32)[
            end - POWER_HISTORY + 1 : end + 1
        ]
        values = np.nan_to_num(values, nan=0.0)
        values = ((values - self.power_mean) / self.power_std).astype(
            np.float32
        )
        inputs = np.stack(
            [
                values,
                np.full_like(values, self.label, dtype=np.float32),
            ],
            axis=-1,
        )
        return (inputs + self.power_encoding).astype(np.float32)

    def add_weather_encoding(
        self,
        past: np.ndarray,
        future: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        spatial = self.relative_bias[None, :, :]
        past = past + self.weather_past_encoding[:, None, :] + spatial
        future = future + self.weather_future_encoding[:, None, :] + spatial
        return past.astype(np.float32), future.astype(np.float32)


class ForecastWeather:
    def __init__(self, site: PublicSite):
        self.site = site

    @functools.lru_cache(maxsize=32)
    def block(self, base_key: str) -> np.ndarray:
        path = (
            self.site.data_root
            / "weather_infer"
            / str(self.site.station_id)
            / f"{base_key}_{self.site.station_id}.npy"
        )
        if not path.exists():
            raise FileNotFoundError(path)
        weather = repair_cumulative_weather(
            np.load(path, allow_pickle=False).astype(np.float32, copy=True)
        )
        weather = (
            weather - self.site.weather_mean[None, None, :]
        ) / self.site.weather_std[None, None, :]
        weather *= self.site.weather_mask[None, None, :]
        return np.nan_to_num(
            weather,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)

    @functools.lru_cache(maxsize=2048)
    def at_hour(self, time_bjt: pd.Timestamp) -> np.ndarray:
        base, offset = available_forecast_base(time_bjt)
        key = f"{base:%Y-%m-%d_%H}:00:00"
        try:
            block = self.block(key)
            if 0 <= offset < len(block):
                return block[offset]
        except FileNotFoundError:
            pass
        return self.site.reference_weather_at(time_bjt)

    def inputs(
        self,
        issue_time: pd.Timestamp,
    ) -> tuple[np.ndarray, np.ndarray]:
        past_times = pd.date_range(
            issue_time.floor("h") - pd.Timedelta(hours=WEATHER_HISTORY),
            periods=WEATHER_HISTORY,
            freq="1h",
        )
        past = np.stack([self.at_hour(time) for time in past_times])
        base, offset = available_forecast_base(issue_time)
        key = f"{base:%Y-%m-%d_%H}:00:00"
        future = self.block(key)[offset : offset + WEATHER_FUTURE]
        if len(future) != WEATHER_FUTURE:
            raise ValueError(
                f"station {self.site.station_id}: incomplete forecast weather"
            )
        return self.site.add_weather_encoding(past, future)


def task_indices(task: str) -> np.ndarray:
    if task == "full":
        return np.arange(FORECAST_STEPS, dtype=np.int64)
    if task == "ultra_short":
        return np.arange(16, dtype=np.int64)
    if task == "short":
        return np.arange(63, 159, dtype=np.int64)
    if task == "medium":
        return np.arange(351, 447, dtype=np.int64)
    raise ValueError(task)


def inspect_data(
    data_root: Path,
    model: Path | None = None,
    heads_dir: Path | None = None,
) -> None:
    info = pd.read_csv(data_root / "info.csv")
    counts = info["type"].value_counts().to_dict()
    power_files = len(list((data_root / "active_power_norm").glob("*.csv")))
    training_weather = len(
        list((data_root / "weather_finetune").glob("*/*.npy"))
    )
    forecast_weather = len(
        list((data_root / "weather_infer").glob("*/*.npy"))
    )
    required_aux = (
        data_root / "model_aux" / "weather_normalization.npz",
        data_root / "model_aux" / "relative_bias.npy",
        data_root / "model_aux" / "weather_var_mask.npy",
    )
    if len(info) != 105 or counts != {"solar": 60, "wind": 45}:
        raise ValueError(f"unexpected site inventory: {len(info)} sites, {counts}")
    if power_files != 105:
        raise ValueError(f"expected 105 power files, found {power_files}")
    missing = [str(path) for path in required_aux if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing model auxiliaries: {missing}")
    if training_weather == 0 or forecast_weather == 0:
        raise ValueError("meteorological files are incomplete")

    expected_columns = [
        "station_id",
        "type",
        "type_id",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
    ]
    if list(info.columns) != expected_columns:
        raise ValueError("info.csv has unexpected columns")
    if (
        not info["station_id"].is_unique
        or set(info["station_id"]) != set(range(105))
    ):
        raise ValueError("info.csv must contain unique station IDs 0–104")
    if not info["type"].map({"wind": 0, "solar": 1}).equals(info["type_id"]):
        raise ValueError("info.csv contains inconsistent site types")
    dates = info[["train_start", "train_end", "test_start", "test_end"]].apply(
        pd.to_datetime, errors="raise"
    )
    if dates.isna().any().any() or not (
        (dates["train_start"] <= dates["train_end"])
        & (dates["test_start"] <= dates["test_end"])
    ).all():
        raise ValueError("info.csv contains invalid date ranges")
    if not (dates["train_end"] < dates["test_start"]).all():
        raise ValueError("adaptation and evaluation periods overlap")
    print("[PASS] public data inventory: 105 sites")
    print("[PASS] power and meteorological files are complete")
    print("[PASS] model auxiliary files are complete")
    print("[PASS] info.csv defines valid, non-overlapping data periods")
    if model is not None:
        if not model.is_file():
            raise FileNotFoundError(f"FM file is missing: {model}")
        model_graph = onnx.load_model(str(model), load_external_data=False).graph
        if not model_graph.input or not model_graph.output:
            raise ValueError("FM has an invalid ONNX graph")
        print("[PASS] FM is available in ONNX format")
    if heads_dir is not None:
        expected_heads = {
            f"station_{station_id}.safetensors"
            for station_id in info["station_id"].astype(int)
        }
        available_heads = {path.name for path in heads_dir.glob("*.safetensors")}
        missing_heads = sorted(expected_heads - available_heads)
        extra_heads = sorted(available_heads - expected_heads)
        if missing_heads or extra_heads:
            raise ValueError(
                "forecast-head inventory does not match info.csv: "
                f"missing={missing_heads}, extra={extra_heads}"
            )
        print("[PASS] fine-tuned forecast heads: 105 sites")


def selected_station_ids(args: argparse.Namespace) -> list[int]:
    available = sorted(
        pd.read_csv(args.data_root / "info.csv")["station_id"]
        .astype(int)
        .tolist()
    )
    if args.all_sites:
        selected = available
    elif args.station_id:
        unknown = sorted(set(args.station_id) - set(available))
        if unknown:
            raise ValueError(f"unknown station IDs: {unknown}")
        selected = sorted(set(args.station_id))
    else:
        raise ValueError("use --station-id ID or --all-sites")
    return selected[args.shard_index :: args.num_shards]


def patched_model(model: Path, head: Path) -> bytes:
    """Return the FM ONNX graph with one released forecast head applied."""
    if not head.exists():
        raise FileNotFoundError(head)
    graph = onnx.load_model(model, load_external_data=True)
    initializers = {tensor.name: tensor for tensor in graph.graph.initializer}
    head_tensors = load_safetensors(head)
    if not head_tensors:
        raise ValueError(f"forecast head contains no tensors: {head}")
    missing = sorted(set(head_tensors) - set(initializers))
    if missing:
        raise ValueError(f"forecast-head tensors are absent from the FM: {missing}")
    for name, array in head_tensors.items():
        expected = numpy_helper.to_array(initializers[name])
        if expected.shape != array.shape:
            raise ValueError(
                f"{name}: expected shape {expected.shape}, found {array.shape}"
            )
        initializers[name].CopyFrom(
            numpy_helper.from_array(np.asarray(array, dtype=expected.dtype), name)
        )
    return graph.SerializeToString()


def create_session(
    model: Path,
    provider: str,
    head: Path | None = None,
) -> ort.InferenceSession:
    if not model.exists():
        raise FileNotFoundError(model)
    available = ort.get_available_providers()
    requested = (
        "CUDAExecutionProvider" if provider == "cuda" else "CPUExecutionProvider"
    )
    if requested not in available:
        raise RuntimeError(
            f"{requested} is unavailable; installed providers: {available}"
        )
    if provider == "cuda" and hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.log_severity_level = 3
    model_source: str | bytes = str(model)
    if head is not None:
        model_source = patched_model(model, head)
    return ort.InferenceSession(
        model_source,
        sess_options=options,
        providers=[requested],
    )


def run_forecast(args: argparse.Namespace) -> None:
    if args.model is None:
        raise ValueError("--model is required for forecast mode")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    station_ids = selected_station_ids(args)
    if args.issue_time is None:
        info = (
            pd.read_csv(args.data_root / "info.csv")
            .set_index("station_id")
            .loc[station_ids]
        )
        start = pd.to_datetime(info["test_start"]).max()
        end = pd.to_datetime(info["test_end"]).min()
        issue_time = start.ceil("D") + pd.Timedelta(hours=8)
        if issue_time + pd.Timedelta(days=12) <= end:
            issue_time += pd.Timedelta(days=7)
        last_target = (
            issue_time + (int(task_indices(args.task)[-1]) + 1) * POWER_STEP
        )
        if last_target > end:
            raise ValueError(
                "Selected sites have no common forecast interval; "
                "use evaluate --all-sites or specify one station"
            )
        print(f"selected issue time: {issue_time}", flush=True)
    else:
        issue_time = pd.Timestamp(args.issue_time)
    if args.task in {"short", "medium"} and (
        issue_time.hour != 8 or issue_time.minute != 0
    ):
        raise ValueError("short and medium forecasts must be issued at 08:00")

    output_indices = task_indices(args.task)
    if args.model_state == "fine-tuned" and args.heads_dir is None:
        raise ValueError("--heads-dir is required for fine-tuned forecasting")
    shared_session = (
        create_session(args.model, args.provider)
        if args.model_state == "zero-shot"
        else None
    )
    rows: list[pd.DataFrame] = []

    for start in range(0, len(station_ids), args.batch_size):
        batch_ids = station_ids[start : start + args.batch_size]
        sites: list[PublicSite] = []
        weather_past: list[np.ndarray] = []
        weather_future: list[np.ndarray] = []
        power_past: list[np.ndarray] = []
        for station_id in batch_ids:
            site = PublicSite(args.data_root, station_id)
            past, future = ForecastWeather(site).inputs(issue_time)
            sites.append(site)
            weather_past.append(past)
            weather_future.append(future)
            power_past.append(site.power_input(issue_time))

        if shared_session is not None:
            forecasts = shared_session.run(
                ["forecast"],
                {
                    "input_weather_past": np.stack(weather_past),
                    "input_weather_future": np.stack(weather_future),
                    "input_power_past": np.stack(power_past),
                },
            )[0]
        else:
            assert args.heads_dir is not None
            forecasts = []
            for batch_index, site in enumerate(sites):
                session = create_session(
                    args.model,
                    args.provider,
                    args.heads_dir / f"station_{site.station_id}.safetensors",
                )
                site_forecast = session.run(
                    ["forecast"],
                    {
                        "input_weather_past": weather_past[batch_index][None, ...],
                        "input_weather_future": weather_future[batch_index][None, ...],
                        "input_power_past": power_past[batch_index][None, ...],
                    },
                )[0][0]
                forecasts.append(site_forecast)
                del session
            forecasts = np.stack(forecasts)

        target_times = pd.DatetimeIndex(
            [
                issue_time + (int(index) + 1) * POWER_STEP
                for index in output_indices
            ]
        )
        for batch_index, site in enumerate(sites):
            prediction = (
                forecasts[batch_index, output_indices] * site.power_std
                + site.power_mean
            )
            prediction = np.maximum(prediction, 0.0)
            if site.label == 1:
                night = (target_times.hour >= 19) | (target_times.hour <= 7)
                prediction[night] = 0.0
            rows.append(
                pd.DataFrame(
                    {
                        "station_id": site.station_id,
                        "technology": str(site.info["type"]),
                        "task": args.task,
                        "model_state": args.model_state,
                        "issue_time_bjt": issue_time,
                        "target_time_bjt": target_times,
                        "model_step": output_indices + 1,
                        "observed_capacity_normalized": site.power.reindex(
                            target_times
                        ).to_numpy(np.float32),
                        "forecast_capacity_normalized": prediction,
                    }
                )
            )
        print(f"completed {min(start + len(batch_ids), len(station_ids))}"
              f"/{len(station_ids)} sites", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = pd.concat(rows, ignore_index=True)
    result.to_csv(args.output, index=False)
    print(f"wrote {len(result):,} forecasts to {args.output}")


def plot_forecast_comparison(args: argparse.Namespace) -> None:
    if args.zero_shot_forecast is None or args.fine_tuned_forecast is None:
        raise ValueError(
            "--zero-shot-forecast and --fine-tuned-forecast are required for plot mode"
        )
    zero = pd.read_csv(args.zero_shot_forecast)
    tuned = pd.read_csv(args.fine_tuned_forecast)
    required = {
        "station_id",
        "technology",
        "task",
        "issue_time_bjt",
        "target_time_bjt",
        "observed_capacity_normalized",
        "forecast_capacity_normalized",
    }
    for label, frame in (("zero-shot", zero), ("fine-tuned", tuned)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{label} forecast is missing columns: {sorted(missing)}")
        if frame["station_id"].nunique() != 1:
            raise ValueError(f"{label} forecast must contain exactly one station")

    keys = [
        "station_id",
        "technology",
        "task",
        "issue_time_bjt",
        "target_time_bjt",
        "observed_capacity_normalized",
    ]
    merged = zero[keys + ["forecast_capacity_normalized"]].merge(
        tuned[keys + ["forecast_capacity_normalized"]],
        on=keys,
        how="inner",
        suffixes=("_zero_shot", "_fine_tuned"),
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("the two forecast files do not share any target times")
    merged["target_time_bjt"] = pd.to_datetime(merged["target_time_bjt"])
    merged = merged.sort_values("target_time_bjt")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=220)
    ax.plot(
        merged["target_time_bjt"],
        merged["observed_capacity_normalized"],
        color="#202020",
        linewidth=2.0,
        label="Observed power",
    )
    ax.plot(
        merged["target_time_bjt"],
        merged["forecast_capacity_normalized_zero_shot"],
        color="#3377B5",
        linewidth=1.5,
        label="Zero-shot forecast",
    )
    ax.plot(
        merged["target_time_bjt"],
        merged["forecast_capacity_normalized_fine_tuned"],
        color="#C4443D",
        linewidth=1.5,
        label="Fine-tuned forecast",
    )
    first = merged.iloc[0]
    ax.set_title(
        f"Station {int(first['station_id'])} · {first['technology']} · "
        f"{first['task']} · issued {first['issue_time_bjt']}"
    )
    ax.set_ylabel("Capacity-normalized power")
    ax.set_xlabel("Target time (BJT)")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    fig.autofmt_xdate(rotation=20, ha="right")
    fig.tight_layout()
    args.plot_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot_output, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote forecast comparison to {args.plot_output}")


def evaluation_plan(
    site: PublicSite,
    task: str,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    test_days = pd.date_range(
        site.test_start.floor("D"),
        site.test_end.floor("D"),
        freq="1D",
    )
    if task == "ultra_short":
        issue_times = pd.date_range(
            site.test_start - POWER_STEP,
            site.test_end - pd.Timedelta(hours=4),
            freq="4h",
        )
    elif task == "short":
        issue_times = pd.DatetimeIndex(
            [
                day - pd.Timedelta(days=1) + pd.Timedelta(hours=8)
                for day in test_days
            ]
        )
    elif task == "medium":
        issue_times = pd.DatetimeIndex(
            [
                day - pd.Timedelta(days=4) + pd.Timedelta(hours=8)
                for day in test_days
            ]
        )
    else:
        raise ValueError(task)
    return issue_times, task_indices(task)


def evaluate_site_task(
    session: ort.InferenceSession,
    site: PublicSite,
    task: str,
    batch_size: int,
) -> dict[str, object]:
    issue_times, output_indices = evaluation_plan(site, task)
    weather = ForecastWeather(site)
    count = 0
    absolute_error = 0.0
    squared_error = 0.0
    truth_sum = 0.0
    prediction_sum = 0.0
    truth_squared_sum = 0.0
    prediction_squared_sum = 0.0
    cross_product_sum = 0.0

    for start in range(0, len(issue_times), batch_size):
        batch_issues = issue_times[start : start + batch_size]
        weather_past: list[np.ndarray] = []
        weather_future: list[np.ndarray] = []
        power_past: list[np.ndarray] = []
        for issue_time in batch_issues:
            past, future = weather.inputs(issue_time)
            weather_past.append(past)
            weather_future.append(future)
            power_past.append(site.power_input(issue_time))

        forecasts = session.run(
            ["forecast"],
            {
                "input_weather_past": np.stack(weather_past),
                "input_weather_future": np.stack(weather_future),
                "input_power_past": np.stack(power_past),
            },
        )[0][:, output_indices]

        for batch_index, issue_time in enumerate(batch_issues):
            target_times = pd.DatetimeIndex(
                [
                    issue_time + (int(index) + 1) * POWER_STEP
                    for index in output_indices
                ]
            )
            prediction = (
                forecasts[batch_index] * site.power_std + site.power_mean
            )
            prediction = np.maximum(prediction, 0.0)
            if site.label == 1:
                night = (target_times.hour >= 19) | (target_times.hour <= 7)
                prediction[night] = 0.0
            truth = site.power.reindex(target_times).to_numpy(np.float32)
            if site.label == 0:
                persistence = np.full(len(target_times), site.power.get(issue_time, np.nan))
            else:
                lag_hours = {"ultra_short": 24, "short": 48, "medium": 120}[task]
                persistence = site.power.reindex(target_times - pd.Timedelta(hours=lag_hours)).to_numpy(np.float32)
            valid = np.isfinite(prediction) & np.isfinite(truth) & np.isfinite(persistence)
            if np.any(valid):
                prediction_valid = prediction[valid].astype(np.float64)
                truth_valid = truth[valid].astype(np.float64)
                error = prediction_valid - truth_valid
                count += int(error.size)
                absolute_error += float(np.abs(error).sum())
                squared_error += float(np.dot(error, error))
                truth_sum += float(truth_valid.sum())
                prediction_sum += float(prediction_valid.sum())
                truth_squared_sum += float(np.dot(truth_valid, truth_valid))
                prediction_squared_sum += float(
                    np.dot(prediction_valid, prediction_valid)
                )
                cross_product_sum += float(
                    np.dot(truth_valid, prediction_valid)
                )

    if count == 0:
        raise ValueError(f"station {site.station_id} {task}: no valid targets")
    truth_variation = truth_squared_sum - truth_sum * truth_sum / count
    prediction_variation = (
        prediction_squared_sum - prediction_sum * prediction_sum / count
    )
    covariance = cross_product_sum - truth_sum * prediction_sum / count
    r2 = (
        1.0 - squared_error / truth_variation
        if truth_variation > 0.0
        else float("nan")
    )
    corr_denominator = np.sqrt(
        max(truth_variation, 0.0) * max(prediction_variation, 0.0)
    )
    corr = (
        covariance / corr_denominator
        if corr_denominator > 0.0
        else float("nan")
    )
    return {
        "station_id": site.station_id,
        "technology": str(site.info["type"]),
        "task": task,
        "observations": count,
        "MAE_percent": float(absolute_error / count * 100.0),
        "RMSE_percent": float(np.sqrt(squared_error / count) * 100.0),
        "R2": float(r2),
        "Corr": float(corr),
    }


def evaluate_model(args: argparse.Namespace) -> None:
    if args.model is None:
        raise ValueError("--model is required for evaluate mode")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.model_state == "fine-tuned" and args.heads_dir is None:
        raise ValueError("--heads-dir is required for fine-tuned evaluation")
    station_ids = selected_station_ids(args)
    shared_session = (
        create_session(args.model, args.provider)
        if args.model_state == "zero-shot"
        else None
    )
    metric_dir = args.output_dir / "site_metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)

    for position, station_id in enumerate(station_ids, start=1):
        output = metric_dir / f"station_{station_id}.csv"
        if output.exists() and not args.overwrite:
            completed = pd.read_csv(output)
            expected_columns = {
                "station_id",
                "technology",
                "task",
                "observations",
            }
            expected_columns.update(SITE_METRIC_COLUMNS)
            if (
                set(completed["task"]) == {"ultra_short", "short", "medium"}
                and expected_columns.issubset(completed.columns)
            ):
                print(
                    f"station {station_id}: already evaluated "
                    f"({position}/{len(station_ids)})",
                    flush=True,
                )
                continue
        if shared_session is None:
            assert args.heads_dir is not None
            session = create_session(
                args.model,
                args.provider,
                args.heads_dir / f"station_{station_id}.safetensors",
            )
        else:
            session = shared_session
        site = PublicSite(args.data_root, station_id)
        rows = [
            evaluate_site_task(session, site, task, args.batch_size)
            for task in ("ultra_short", "short", "medium")
        ]
        for row in rows:
            row["model_state"] = args.model_state
        pd.DataFrame(rows).to_csv(output, index=False)
        values = ", ".join(
            f"{row['task']} RMSE={row['RMSE_percent']:.3f}%" for row in rows
        )
        print(
            f"station {station_id}: {values} "
            f"({position}/{len(station_ids)})",
            flush=True,
        )
        if shared_session is None:
            del session
            gc.collect()

    if args.num_shards == 1 and args.all_sites:
        summarize_evaluation(args, require_complete=True)
    elif args.num_shards > 1:
        print(
            "This shard is complete. After all shards finish, run summarize "
            "with the same --data-root and --output-dir.",
            flush=True,
        )


def summarize_evaluation(
    args: argparse.Namespace,
    require_complete: bool = True,
) -> None:
    metric_dir = args.output_dir / "site_metrics"
    files = sorted(metric_dir.glob("station_*.csv"))
    if not files:
        raise FileNotFoundError(f"no site metrics under {metric_dir}")
    site_metrics = pd.concat(
        [pd.read_csv(path) for path in files],
        ignore_index=True,
    )
    duplicated = site_metrics.duplicated(["station_id", "task"])
    if duplicated.any():
        raise ValueError("duplicate station-task rows in evaluation output")

    info = pd.read_csv(args.data_root / "info.csv")
    expected_sites = set(info["station_id"].astype(int))
    evaluated_sites = set(site_metrics["station_id"].astype(int))
    if require_complete and evaluated_sites != expected_sites:
        missing = sorted(expected_sites - evaluated_sites)
        extra = sorted(evaluated_sites - expected_sites)
        raise ValueError(
            f"evaluation is incomplete: missing={missing}, extra={extra}"
        )
    counts = site_metrics.groupby("station_id")["task"].nunique()
    if not (counts == 3).all():
        raise ValueError("each evaluated station must contain all three tasks")
    missing_metrics = set(SITE_METRIC_COLUMNS) - set(site_metrics.columns)
    if missing_metrics:
        raise ValueError(f"site metrics are missing columns: {missing_metrics}")

    site_metrics = site_metrics.sort_values(
        ["technology", "station_id", "task"]
    ).reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    site_metrics.to_csv(args.output_dir / "site_metrics.csv", index=False)

    rows: list[dict[str, object]] = []
    for technology in ("wind", "solar"):
        technology_rows = site_metrics.loc[
            site_metrics["technology"] == technology
        ]
        if technology_rows.empty:
            if require_complete:
                raise ValueError(f"evaluation contains no {technology} sites")
            continue
        ultra_short = technology_rows.loc[
            technology_rows["task"] == "ultra_short"
        ]
        ultra_summary: dict[str, object] = {
            "technology": technology,
            "task": "ultra_short",
            "sites": ultra_short["station_id"].nunique(),
        }
        for metric in SITE_METRIC_COLUMNS:
            ultra_summary[metric] = ultra_short[metric].mean()
        rows.append(ultra_summary)

        short_medium_rows = technology_rows.loc[
            technology_rows["task"].isin(["short", "medium"])
        ]
        short_medium_summary: dict[str, object] = {
            "technology": technology,
            "task": "short_medium",
            "sites": short_medium_rows["station_id"].nunique(),
        }
        for metric in SITE_METRIC_COLUMNS:
            pivoted = short_medium_rows.pivot(
                index="station_id", columns="task", values=metric
            )
            if set(pivoted.columns) != {"short", "medium"}:
                raise ValueError(
                    f"{technology}: incomplete short/medium {metric} values"
                )
            site_values = pivoted[["short", "medium"]].dropna().mean(axis=1)
            short_medium_summary[metric] = site_values.mean()
        rows.append(short_medium_summary)

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    states = sorted(site_metrics.get("model_state", pd.Series(dtype=str)).dropna().unique())
    state = states[0] if len(states) == 1 else args.model_state
    print(f"\nSite-first {state} metrics")
    print(
        summary.to_string(
            index=False, float_format=lambda value: f"{value:.3f}"
        )
    )


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be within --num-shards")
    if args.mode == "inspect":
        inspect_data(args.data_root, args.model, args.heads_dir)
    elif args.mode == "forecast":
        run_forecast(args)
    elif args.mode == "evaluate":
        evaluate_model(args)
    elif args.mode == "plot":
        plot_forecast_comparison(args)
    else:
        summarize_evaluation(args)


if __name__ == "__main__":
    main()
