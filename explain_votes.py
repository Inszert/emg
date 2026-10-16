"""
================================================================================
 BUILD THE DECISION PANEL FOR THE MODEL THAT IS ALREADY SAVED
================================================================================

 The sweep writes `models/gesture_panel.joblib` itself, so after a normal
 retrain there is nothing to do here. This script exists for the other case:
 a model that was trained before the panel existed, or a panel you deleted,
 where re-running a three-hour sweep just to get the view back would be
 absurd.

 It reads the setup out of `models/gesture_model.joblib` -- window length,
 overlap, representation, feature set, exclusions -- rebuilds exactly that
 dataset, and runs `train_model.build_decision_panel()` on it. One definition
 of what a panel is, shared with the trainer.

 It does NOT touch `gesture_model.joblib`. The deployed model is left exactly
 as it was; only the side file is written.

 Run:
     python explain_votes.py                    # panel for the saved model
     python explain_votes.py --json             # also a JSON copy of the trail
     python explain_votes.py --out models/x.joblib

 The --json copy is what the browser view of the same data reads; it is not
 used by the application.
================================================================================
"""

import argparse
import json
import os
import sys
import time

import numpy as np

import emg_dataset_tool as T
import train_model as TM

try:
    import joblib
except ImportError:                                          # pragma: no cover
    print("joblib is required:  python -m pip install joblib")
    sys.exit(1)


def setup_from_payload(payload: dict) -> dict:
    """
    The winner's setup, read off the saved model.

    Shaped like a sweep row, because that is what `build_decision_panel()`
    takes. Anything missing is a model from before that field was recorded,
    and is reported rather than guessed.
    """
    stats = dict(payload.get("stats") or {})
    missing = [k for k in ("model_name", "overlap", "excluded")
               if stats.get(k) is None]
    if missing:
        raise SystemExit(
            "This model does not record " + ", ".join(missing) +
            ", so the panel cannot be built for the same setup it used.\n"
            "Retrain (python train_model.py) -- the sweep writes the panel "
            "itself.")
    return {
        "model": stats["model_name"],
        "window_s": float(payload.get("window_s", stats.get("window_s"))),
        "overlap": float(stats["overlap"]),
        "representation": payload.get("representation",
                                      stats.get("representation")),
        "feature_set": payload.get("feature_set", stats.get("feature_set")),
        "freq_features": bool(payload.get("freq_features", True)),
        "exclude": stats["excluded"],
    }


def to_json(panel: dict, labels, uids) -> dict:
    """The panel as plain JSON, for the browser view of the same numbers."""
    trail = panel["trail"]
    order = list(panel["member_order"])
    vote = panel["vote_name"]
    proba = trail["proba"]
    truth = np.asarray(trail["truth"])
    chan = trail["chan"]
    positive = min(1, len(labels) - 1)

    windows = []
    for row in range(truth.shape[0]):
        entry = {
            "r": int(np.asarray(trail["rec"])[row]),
            "t": int(truth[row]),
            "f": int(np.asarray(trail["fold"])[row]),
            "p": [round(float(np.asarray(proba[n])[row, positive]), 4)
                  for n in order],
        }
        if vote is not None:
            entry["v"] = round(float(np.asarray(proba[vote])[row, positive]), 4)
        if chan is not None:
            entry["c"] = [round(float(v), 1) for v in np.asarray(chan)[row]]
        windows.append(entry)

    models = []
    for name in order + ([vote] if vote else []):
        stats = dict(panel["cv"].get(name, {}))
        models.append({
            "name": name,
            "is_vote": name == vote,
            "kind": "voting_soft" if name == vote else "",
            "scaler": "none",
            "params": {},
            "window_accuracy": stats.get("window_accuracy"),
            "recording_accuracy": stats.get("recording_accuracy"),
            "confusion": stats.get("confusion_matrix"),
            "mean_confidence": stats.get("mean_confidence"),
        })

    recordings = []
    for entry in panel["recordings"]:
        row = dict(entry)
        row["models"] = {
            name: {"vote": value["vote"],
                   "n_for_truth": value["n_for_truth"],
                   "correct": value["correct"],
                   "mean_p": None}
            for name, value in entry["models"].items()
        }
        recordings.append(row)

    return {
        "generated_iso": panel["created_iso"],
        "dataset_root": os.path.join(T.PROJECT_DIR, T.DATA_FOLDER),
        "setup": panel["setup"],
        "labels": list(labels),
        "positive_class": labels[positive],
        "protocol": dict(panel["protocol"],
                         subject_independent=None,
                         subject_note="one subject only; no cross-subject "
                                      "claim possible"),
        "n_recordings": panel["n_recordings"],
        "n_windows": panel["n_windows"],
        "member_names": order,
        "vote_name": vote,
        "vote_mode": "soft",
        "fusion": panel["fusion"],
        "models": models,
        "recordings": recordings,
        "windows": windows,
    }


def main(argv=None):
    T.force_utf8_console()
    parser = argparse.ArgumentParser(
        description="Build the decision panel for the saved model")
    parser.add_argument("--model-path", default=None,
                        help="the model to explain (default: MODEL_PATH)")
    parser.add_argument("--root", default=None,
                        help="dataset root (default: the configured one)")
    parser.add_argument("--out", default=None,
                        help="where to write the panel (default: beside the "
                             "model, as gesture_panel.joblib)")
    parser.add_argument("--json", action="store_true",
                        help="also write models/reports/decisions.json")
    parser.add_argument("--quick", action="store_true",
                        help="halve the candidate roster, as --quick does in "
                             "the trainer")
    args = parser.parse_args(argv)

    model_path = args.model_path or os.path.join(T.PROJECT_DIR, T.MODEL_PATH)
    root = args.root or os.path.join(T.PROJECT_DIR, T.DATA_FOLDER)
    if not os.path.exists(model_path):
        raise SystemExit(f"no model at {model_path}. Train one first:\n"
                         "    python train_model.py")

    started = time.time()
    print("=" * 78)
    print(" DECISION PANEL")
    print("=" * 78)
    payload = joblib.load(model_path)
    best = setup_from_payload(payload)
    print(f"  model        : {best['model']}  "
          f"(saved {payload.get('created_iso', '?')})")
    print(f"  setup        : {best['window_s']} s, overlap {best['overlap']}, "
          f"{best['representation']}/{best['feature_set']}, "
          f"exclude={best['exclude']}")

    spec, spec_source = TM.resolve_spec(
        payload.get("model_spec_source")
        if os.path.exists(str(payload.get("model_spec_source") or ""))
        else T.model_spec_path(model_path))
    print(f"  spec         : {spec_source}")

    exclude = TM.EXCLUDE_SETS.get(best["exclude"], ())
    recs, dropped = TM.load_recordings(root, exclude_codes=exclude)
    print(f"  recordings   : {len(recs)} kept, {len(dropped)} dropped")

    # The calibration mode comes from the PAYLOAD, not from the current
    # configuration: this script rebuilds the panel for the model that is
    # already saved, so it has to build the features that model was fitted
    # on. Reading CALIBRATION_MODE here would silently produce a panel on a
    # different scale from the model it is supposed to explain.
    cal_mode = (payload.get("calibration") or {}).get("mode", "off")
    cal_table = T.calibration_table(root, cal_mode)
    print(f"  calibration  : mode {cal_mode}"
          + (f", wearers {', '.join(sorted(cal_table))}" if cal_table else ""))
    if cal_mode != "off" and not cal_table:
        print("  WARNING: the model was trained with calibration but no "
              "usable calibration is on disk now. The panel will be built "
              "uncalibrated and the view will say it does not match.")

    # Same dedup setting the trainer used, so the panel is fitted on the data
    # the saved model was scored on rather than on a leakier superset.
    built = TM.build_dataset(recs, best["window_s"], best["overlap"],
                             best["representation"], best["freq_features"],
                             best["feature_set"],
                             dedupe=TM.DEDUPE_WINDOWS,
                             calibration=cal_table)
    if built is None:
        raise SystemExit("that setup produced no windows; nothing to do.")
    X, y, groups, labels, meta = built
    print(f"  windows      : {X.shape[0]} x {X.shape[1]} features, "
          f"labels {labels}")

    if list(labels) != list(payload.get("labels", labels)):
        print(f"  WARNING: the dataset now has labels {labels} but the model "
              f"was trained on {payload.get('labels')}. The panel will not "
              f"match the model and the view will say so.")

    print("\nScoring every candidate on that setup...")
    panel = TM.build_decision_panel(spec, args.quick, X, y, groups, labels,
                                    best, meta,
                                    calibration_mode=cal_mode)
    if panel is None:
        raise SystemExit("the panel could not be built (too few recordings "
                         "or no usable candidates).")

    out = args.out or T.panel_path(model_path)
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    staging = out + ".partial"
    joblib.dump(panel, staging)
    os.replace(staging, out)
    print(f"\n  wrote {os.path.relpath(out, T.PROJECT_DIR)} "
          f"({os.path.getsize(out) / 1e6:.1f} MB)")

    if args.json:
        report_dir = os.path.join(os.path.dirname(os.path.abspath(out)),
                                  "reports")
        os.makedirs(report_dir, exist_ok=True)
        dest = os.path.join(report_dir, "decisions.json")
        with open(dest, "w", encoding="utf-8") as handle:
            json.dump(to_json(panel, labels, groups), handle)
        print(f"  wrote {os.path.relpath(dest, T.PROJECT_DIR)} "
              f"({os.path.getsize(dest) / 1e6:.2f} MB)")

    print("\n  --- every candidate, same setup, out of fold ---")
    for name in panel["member_order"] + ([panel["vote_name"]]
                                         if panel["vote_name"] else []):
        stats = panel["cv"][name]
        mark = "  *" if name == panel["winner"] else ""
        print(f"    {name:<20} window {stats['window_accuracy'] * 100:6.2f}%  "
              f"recording {stats['recording_accuracy'] * 100:6.2f}%  "
              f"conf {stats['mean_confidence']:.3f}{mark}")
    print(f"\n  * = the model that actually predicts "
          f"({panel['winner']}); the rest are shown for comparison only.")
    print(f"  done in {time.time() - started:.1f} s. Open MODEL STATS -> "
          f"DECISION TRAIL in the app.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
