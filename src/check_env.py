"""Check that the project environment is set up.

Run from the repo root:  python src/check_env.py
"""
import importlib
import shutil
import subprocess
import sys

PACKAGES = [
    "numpy", "pandas", "scipy", "sklearn", "xgboost", "shap", "yaml", "tqdm",
    "torch", "rtdl_revisiting_models", "pydicom", "nibabel", "nilearn",
    "neuroCombat", "matplotlib",
]


def main() -> int:
    ok = True
    print(f"Python {sys.version.split()[0]}")
    if sys.version_info[:2] not in [(3, 11), (3, 12)]:
        print("  WARNING: expected Python 3.11 or 3.12")
        ok = False

    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            print(f"  ok   {name} {getattr(mod, '__version__', '')}")
        except Exception as e:  # noqa: BLE001 - report any import failure
            print(f"  FAIL {name}: {e}")
            ok = False

    import torch
    if torch.cuda.is_available():
        x = torch.randn(256, 256, device="cuda")
        _ = (x @ x).sum().item()  # fails here if the GPU arch isn't supported by this build
        print(f"  GPU  {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda}) - works")
    else:
        print("  GPU  not available (CPU only); fine for tabular models, slow for MRI")

    exe = shutil.which("dcm2niix")
    if exe:
        out = subprocess.run([exe, "-v"], capture_output=True, text=True)
        print(f"  ok   dcm2niix {(out.stdout or out.stderr).strip().splitlines()[0]}")
    else:
        print("  FAIL dcm2niix not on PATH (activate the venv first)")
        ok = False

    print("\nAll good." if ok else "\nSome checks failed (see above).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
