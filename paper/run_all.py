"""Reproduce every table, figure and statistical test of the paper from results/ and data/ (about 10 minutes on a laptop)."""
import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402,F401  (creates outputs/{tables,figures,stats})

for script in ["fig1_mechanism.py", "figS1_units.py", "main_tests.py", "hier_bootstrap.py", "tables.py", "exploratory_tests.py",
               "uncertainty_tests.py"]:
    t = time.time()
    print(f"== {script}", flush=True)
    runpy.run_path(str(HERE / script), run_name="__main__")
    print(f"   {time.time() - t:.0f}s", flush=True)
