# Public 105-site inference and evaluation

The program evaluates the shared FM (`zero-shot`) or applies a released
site-specific forecast head (`fine-tuned`). Data contain 45 wind and 60 solar
sites, identified by anonymous IDs 0–104.

## Download and layout

Get all three data parts and `IResFM_release_v1_model.tar.zst` from
[Baidu Netdisk](https://pan.baidu.com/s/1Yf3faf4TXVH62QRDJVutNQ?pwd=ires),
access code `ires`. Keep the three parts in one folder. From the repository root:

```bash
cat /path/to/download/IResFM_release_v1_data.tar.zst.part-* | tar -I zstd -xf - -C public_105
tar -I zstd -xf /path/to/download/IResFM_release_v1_model.tar.zst -C public_105
```

For the alternative [GitHub Release](https://github.com/PussyCat0700/iresfm_main_method/releases)
route, place `iresfm_base.onnx` in `models/` and extract all four head archives
into `heads/`. Both download routes provide the same weights.

```text
public_105/
├── run.py
├── requirements.txt
├── models/iresfm_base.onnx
├── heads/station_<id>.safetensors
└── data/iresfm_open_105/
    ├── info.csv
    ├── active_power_norm/<id>.csv
    ├── weather_finetune/<id>/*.npy
    ├── weather_infer/<id>/*.npy
    └── model_aux/
        ├── weather_normalization.npz
        ├── weather_var_mask.npy
        └── relative_bias.npy
```

## Data fields

`info.csv` records each site's type and adaptation and evaluation periods:

| Field | Meaning |
|---|---|
| `station_id` | Anonymous integer 0–104 |
| `type`, `type_id` | `wind`/0 or `solar`/1 |
| `train_start`, `train_end` | Adaptation interval, inclusive |
| `test_start`, `test_end` | Evaluation interval, inclusive |

Power CSV columns are `time,power`, sampled every 15 minutes in UTC+8.
Power is divided by installed capacity. At model input, the runner additionally
standardizes power using the adaptation-period mean and standard deviation;
forecasts are converted back to capacity-normalized values.

Weather filenames use UTC. Arrays are float32 with shape `[12,64,27]` for
historical blocks and `[131,64,27]` for forecast blocks: hourly time steps,
8×8 spatial cells and 27 channels. The runner uses the supplied per-site
weather statistics, channel masks and relative spatial encodings. Keep these
auxiliaries with their corresponding data and forecast heads.

Original station identifiers, names, coordinates and installed capacities are
not included.

## Commands

Paths default to the layout above and can be overridden using `--data-root`,
`--model` and `--heads-dir`. Outputs are written relative to the working directory.

```bash
python public_105/run.py inspect
python public_105/run.py forecast --station-id 0 --task ultra_short --output outputs/forecast.csv
python public_105/run.py forecast --station-id 0 --model-state fine-tuned --output outputs/forecast_fine_tuned.csv
python public_105/run.py evaluate --all-sites --model-state zero-shot --output-dir outputs/zero_shot
python public_105/run.py evaluate --all-sites --model-state fine-tuned --output-dir outputs/fine_tuned
```

`forecast` accepts `full`, `ultra_short`, `short` and `medium`. Use `--issue-time`
to specify a local time, or omit it for a time within the selected site's test
period. Short and medium forecasts use 08:00 issuance. Forecast CSVs include
observed power where available. Use `evaluate --all-sites` to evaluate each site
over the interval recorded in `info.csv`.

Plot observed, zero-shot and fine-tuned forecasts for the same site and issue time:

```bash
python public_105/run.py plot --zero-shot-forecast outputs/forecast_zero_shot.csv --fine-tuned-forecast outputs/forecast_fine_tuned.csv --plot-output outputs/comparison.png
```

Use a fresh output directory after replacing data or weights: evaluation reuses
completed per-site files. `--overwrite` recomputes them.

## GPU and parallel evaluation

CPU dependencies are listed in `requirements.txt`. To use a compatible NVIDIA GPU:

```bash
pip uninstall -y onnxruntime
pip install "onnxruntime-gpu[cuda,cudnn]==1.22.0"
```

Add `--provider cuda` to forecasting or evaluation. For four GPUs:

```bash
mkdir -p outputs/zero_shot/logs
for SHARD in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=${SHARD} python public_105/run.py evaluate     --all-sites --model-state zero-shot --provider cuda     --num-shards 4 --shard-index ${SHARD} --output-dir outputs/zero_shot     > outputs/zero_shot/logs/shard_${SHARD}.log 2>&1 &
done
wait
python public_105/run.py summarize --output-dir outputs/zero_shot
```

## Metrics and outputs

US evaluates the next 16 quarter-hour leads at four-hour issuance intervals.
Short and medium evaluation use the target day from previous-day and
four-days-earlier 08:00 issuance. SM averages short and medium metrics within
each site, then averages across sites. Other summaries also average site-level
metrics. MAE and RMSE are capacity percentage points; R2 and Corr are dimensionless.

Evaluation uses common finite observations and Persistence availability. Wind
Persistence repeats the latest observation; solar reference lags are 24, 48
and 120 hours for US, short and medium. No negative-R2 site filter is applied.

```text
outputs/zero_shot/
├── site_metrics/station_<id>.csv
├── site_metrics.csv
└── summary.csv
```

Results are computed locally from the released data and weights.
