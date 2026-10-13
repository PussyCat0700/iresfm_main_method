# IresFM

IresFM forecasts wind and solar power across ultra-short, short and medium
horizons. This repository provides:

- [`public_105/`](public_105/): runnable inference and evaluation using a shared
  FM and site-specific fine-tuned heads on 45 wind and 60 solar sites.
- [`pseudocode/`](pseudocode/): methodological descriptions of data generation,
  architecture and training. The [experiment protocols](pseudocode/experiments.md)
  document surrogate fidelity and Fig. 3 settings.
- [`figure_archive/`](figure_archive/): plotting data and code for the paper figures.

## Install

```bash
git clone https://github.com/PussyCat0700/iresfm_main_method.git
cd iresfm_main_method
python -m venv .venv
source .venv/bin/activate
pip install -r public_105/requirements.txt
```

## Download

Download the three data parts (`00`, `01`, `02`) and the model archive from
[Baidu Netdisk](https://pan.baidu.com/s/1Yf3faf4TXVH62QRDJVutNQ?pwd=ires),
access code `ires`. Follow its `README.txt` to extract them into `public_105/`.
The model archive contains the FM in ONNX format and all 105 forecast heads.
Alternatively, obtain the same model and heads from
[GitHub Releases](https://github.com/PussyCat0700/iresfm_main_method/releases).
Large data and weights are downloaded separately from `git clone`.

## Run

```bash
# Check the downloaded files.
python public_105/run.py inspect

# Forecast one site; an issue time is selected within its test period.
python public_105/run.py forecast --station-id 0

# Evaluate all sites, separately for each model state.
python public_105/run.py evaluate --all-sites --model-state zero-shot --output-dir outputs/zero_shot
python public_105/run.py evaluate --all-sites --model-state fine-tuned --output-dir outputs/fine_tuned
```

CPU is the default. GPU setup, plotting, parallel evaluation, data fields and
metric definitions are in [`public_105/README.md`](public_105/README.md).

## License

See [`LICENSE`](LICENSE).
