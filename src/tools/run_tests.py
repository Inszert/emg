"""
Run the project's check scripts one after another and print one line each.

    python tools/run_tests.py                 # every offline suite (no hardware)
    python tools/run_tests.py gui zoo         # only those whose name contains a word
    python tools/run_tests.py --list

A suite passes when its process exits 0 and prints "N/N checks passed". The
hardware suites (test_live*.py) and the retired torch trainer's are NOT run.
Nothing here trains a model except `custom_gesture` (on a temp copy, minutes) and
`nn` (starts the CPU/GPU worker), which are listed last and skipped with --fast.

Output: one line per suite and, at the end, `tests/last_run.json` (what ran, how
long, the pass counts) -- the file docs/VERIFICATION.md quotes.
"""
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TESTS = os.path.join(ROOT, "tests")

#  name -> (script, slow?)
SUITES = [
    ("datalayer", "test_datalayer.py", False),
    ("gui", "test_gui.py", False),
    ("review", "test_review.py", False),
    ("label_on_graph", "test_label_on_graph.py", False),
    ("markers", "test_markers.py", False),
    ("free_take", "test_free_take.py", False),
    ("model_spec", "test_model_spec.py", False),
    ("predict", "test_predict.py", False),
    ("trail", "test_trail.py", False),
    ("calibration", "test_calibration.py", False),
    ("jobs", "test_jobs.py", False),
    ("class_mask", "test_class_mask.py", False),
    ("device", "test_device.py", False),
    ("sweep_cost", "test_sweep_cost.py", False),
    ("filters", "test_filters.py", False),
    ("no_torch", "test_no_torch.py", False),
    ("icon", "test_icon.py", False),
    ("zoo", "test_zoo.py", False),
    ("app_settings", "test_app_settings.py", False),
    ("vote_view", "test_vote_view.py", False),
    ("defaults", "test_defaults.py", False),
    ("nn", "test_nn.py", True),
    ("custom_gesture", "test_custom_gesture.py", True),
]


def run(name, script):
    env = dict(os.environ, PYTHONPATH=ROOT, QT_QPA_PLATFORM="offscreen",
               PYTHONIOENCODING="utf-8")
    t0 = time.time()
    proc = subprocess.run([sys.executable, os.path.join(TESTS, script)],
                          cwd=ROOT, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    out = proc.stdout + proc.stderr
    matches = re.findall(r"(\d+)/(\d+) checks passed", out)
    passed, total = (int(matches[-1][0]), int(matches[-1][1])) if matches \
        else (0, 0)
    if not matches and proc.returncode == 0 and "all passed" in out:
        #  test_icon.py's own format: one "ok  ..." line per check.
        passed = total = sum(1 for ln in out.splitlines()
                             if ln.startswith("ok "))
    fails = [ln for ln in out.splitlines()
             if ln.startswith(("FAIL", "not ok"))]
    return {"suite": name, "script": script, "exit": proc.returncode,
            "passed": passed, "total": total, "seconds": round(time.time() - t0, 1),
            "ok": proc.returncode == 0 and total > 0 and passed == total,
            "failures": fails[:10]}


def main(argv):
    if "--list" in argv:
        for name, script, slow in SUITES:
            print(f"{name:<16}{script}{'   (slow)' if slow else ''}")
        return 0
    fast = "--fast" in argv
    words = [a for a in argv if not a.startswith("--")]
    chosen = [(n, s, sl) for n, s, sl in SUITES
              if (not words or any(w in n for w in words))
              and not (fast and sl)]
    results = []
    for name, script, _slow in chosen:
        r = run(name, script)
        results.append(r)
        print(f"{'PASS' if r['ok'] else 'FAIL'}  {name:<16}{r['passed']:>5}/"
              f"{r['total']:<5}{r['seconds']:>8.1f} s"
              + ("" if r["ok"] else f"   exit {r['exit']}  "
                 + "; ".join(r["failures"][:2])), flush=True)
    #  Only a run of the WHOLE set is "the last run"; a filtered run must not
    #  overwrite it (results/experiments.json and the report read this file).
    name = "last_run_partial.json" if words or fast else "last_run.json"
    with open(os.path.join(TESTS, name), "w", encoding="utf-8") as fh:
        json.dump({"finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "results": results}, fh, indent=1)
    bad = [r for r in results if not r["ok"]]
    print(f"\n{len(results) - len(bad)}/{len(results)} suites passed; "
          f"{sum(r['passed'] for r in results)}/"
          f"{sum(r['total'] for r in results)} checks")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
