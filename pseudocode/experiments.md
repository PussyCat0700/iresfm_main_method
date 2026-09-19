# Experiment protocols

This document records the surrogate-data fidelity definitions and Fig. 3
comparison settings. It describes the experimental design for independent
implementation; the runnable 105-site inference and evaluation program is
[`public_105/run.py`](../public_105/run.py).

## Surrogate fidelity

| Level | Wind conversion | Solar conversion |
|---|---|---|
| 1 | Generic wind-speed and air-density response | Generic solar-geometry and irradiance response |
| 2 | Level 1 plus device-specific hub height and turbine power curves | Level 1 plus module efficiency and inverter properties |
| 3 (base) | Level 2 plus roughness, wake interactions, spatial smoothing and site corrections | Level 2 plus mounting, layout, tilt, azimuth, incidence-angle and spectral effects, module temperature and site losses |

The same selected level applies to wind and solar surrogate sites. Power
series are interpolated to 15-minute resolution. See
[`data_generation.py`](data_generation.py) for the physical conversion workflow.

## Fig. 3 comparisons

Each comparison varies one axis while retaining the other base settings.

| Panel | Varied axis | Settings | Base setting |
|---|---|---|---|
| 3a | Surrogate fidelity | Levels 1, 2, 3 | Level 3 |
| 3b | Initialization and adaptation | Surrogate pretrained; random initialization with frozen backbone; random initialization with full-model training | Surrogate pretrained |
| 3c | Model parameters | Small 36M; Medium 75M; Base 136M; Large 226M | Base 136M |
| 3d | Surrogate sites | 2,500; 5,000; 10,000; 20,000 | 10,000 |
| 3e | History length | 1; 3; 5; 10 days | 5 days |
| 3f | Weather sources | C1–C6 below | C3 |

Weather configurations are cumulative: C1 = ECMWF-HRES; C2 adds FengWu;
C3 adds Pangu; C4 adds GFS; C5 adds FuXi; C6 adds Aurora.

## Documented base protocol

- Surrogate data: 5,000 wind and 5,000 solar sites over 2021, approximately
  2.1 million selected site-time samples.
- Pretraining: one epoch, batch size 128, peak learning rate 1e-5,
  10% warmup followed by cosine decay.
- Repeat each comparison setting with 10 independent random runs and use
  temporally held-out real-site data for evaluation.
- Fig. 3b compares adaptation over 20 epochs; the standard benchmark uses
  one adaptation epoch.
- Curve smoothing uses an exponential moving average with decay 0.8 for
  visualization only; it does not modify the evaluation metrics.

Implementing these experiments requires data construction, model construction,
training and evaluation functions. The corresponding methodological references
are [`data_loader.py`](data_loader.py),
[`model_architecture.py`](model_architecture.py),
[`pretraining.py`](pretraining.py), [`finetuning.py`](finetuning.py), and
[`inference.py`](inference.py).
