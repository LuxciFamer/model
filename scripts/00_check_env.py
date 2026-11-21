"""Phase 0: verify the pixi environment (PyTorch/CUDA + GIS stack)."""
import importlib
import sys

mods = ["numpy", "scipy", "pandas", "sklearn", "rasterio", "geopandas", "shapely",
        "pyproj", "matplotlib", "seaborn", "yaml", "torch"]
ok = True
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"[OK] {m:12s} {getattr(mod, '__version__', '')}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[FAIL] {m}: {e}")

import torch  # noqa: E402

print("Python:", sys.version.split()[0])
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), "| torch CUDA:", torch.version.cuda)
sys.exit(0 if ok else 1)
