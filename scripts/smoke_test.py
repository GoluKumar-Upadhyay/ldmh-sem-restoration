"""Executes every notebook end-to-end on SYNTHETIC data (tiny, 2 epochs) to prove the code runs.
The numbers produced are meaningless - this only validates the pipeline.   python scripts/smoke_test.py"""
import os
import shutil
import sys
import time

os.environ["SEMICON_MODE"] = "smoke"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import nbformat
from nbclient import NotebookClient

shutil.rmtree(os.path.join(ROOT, "results_smoke"), ignore_errors=True)
nbs = sorted(f for f in os.listdir(os.path.join(ROOT, "notebooks")) if f.endswith(".ipynb"))
failed = []
for f in nbs:
    t = time.time()
    nb = nbformat.read(os.path.join(ROOT, "notebooks", f), as_version=4)
    try:
        NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": os.path.join(ROOT, "notebooks")}}).execute()
        print(f"OK    {f}  ({time.time() - t:.0f}s)", flush=True)
    except Exception as e:
        failed.append(f)
        print(f"FAIL  {f}: {type(e).__name__}: {str(e)[-1500:]}", flush=True)
        break
print("\nALL NOTEBOOKS PASSED" if not failed else f"\nFAILED: {failed}")
sys.exit(1 if failed else 0)
