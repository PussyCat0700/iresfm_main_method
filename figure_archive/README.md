# Figure plotting

Requires Python 3.11 or newer.

```bash
cd iresfm_main_method/figure_archive
python -m pip install -r requirements.txt
python plot.py
```

The command writes all 21 figures as PNG files to `figures/`.

To generate selected figures or use another output directory:

```bash
python plot.py --figure fig2b --figure si8
python plot.py --figure ed3 --output /path/to/output
```

Available figure names: `fig2a`–`fig2d`, `fig3a`–`fig3f`, `fig4`, `ed1`, `ed3`, and `si2`–`si9`.
