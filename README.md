# viz3D — COLMAP Plotly Viewer

Minimal Plotly viewer for **COLMAP** reconstructions (cameras + sparse points).  
Works directly with COLMAP’s `read_write_model.py` — **no other SfM deps**.

---

## Quick start

### 1) Install
```bash
pip install -r requirements.txt
# or:
pip install "numpy>=1.26.4" "plotly>=5.20.0"
```

### 2) Run the example
```bash
python run_viz3D.py

```

By default it loads the sample model in `DISK_LightGlue/` and opens an interactive Plotly window.

---
### Repo layout

```
.
├─ viz3D.py               # the viewer (points + frusta)
├─ run_viz3D.py           # example runner script
├─ requirements.txt
├─ DISK_LightGlue/        # example 1: small COLMAP model (cameras.*, images.*, points3D.*)
├─ SP_SuperGlue/          # example 2: small COLMAP model (cameras.*, images.*, points3D.*)
└─ read_write_model.py    # from COLMAP (not bundled upstream)
```
