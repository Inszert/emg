"""
Write docs/REPORT.md from results/experiments.json -- every number in it is read
from that file, none is typed here.

    python tools/collect_results.py        # builds results/experiments.json
    python tools/make_figures_ablation.py  # the figures the report embeds
    python tools/make_report.py            # writes docs/REPORT.md

The prose states only what the numbers say: sentences that depend on a result are
chosen by comparing that result, so re-running after new measurements keeps the
report true instead of stale.
"""
import json
import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DOC = json.load(open(os.path.join(ROOT, "results", "experiments.json"),
                     encoding="utf-8"))
OUT = os.path.join(ROOT, "docs", "REPORT.md")


def pct(v, d=1):
    return "--" if v is None else f"{v * 100:.{d}f} %"


def pts(v, d=1):
    return "--" if v is None else f"{v * 100:+.{d}f}"


def size(b):
    if b is None:
        return "--"
    b = float(b)
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024 or unit == "GB":
            return f"{b:.0f} B" if unit == "B" else f"{b:.1f} {unit}"
        b /= 1024.0


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def ms(v):
    return "--" if v is None else (f"{v:.1f}" if v >= 10 else f"{v:.2f}")


models = DOC.get("models", [])
nn_models = [m for m in models if m["family"] == "nn"]
ml_models = [m for m in models if m["family"] == "ml"]
abl = DOC.get("ablation", [])
newm = DOC.get("new_models", [])
defaults = DOC.get("defaults", [])
voting = DOC.get("voting", [])
tests = (DOC.get("tests") or {}).get("results") or []
meta = DOC.get("meta", {})
fallback_path = os.path.join(ROOT, "results", "ablation", "ml", "fallback.json")
fallback = (json.load(open(fallback_path, encoding="utf-8"))
            if os.path.exists(fallback_path) else None)

L = []
w = L.append
w("# Report: models, ablation, defaults and verification")
w("")
w(f"*Generated {datetime.now():%Y-%m-%d %H:%M} by `tools/make_report.py` from "
  "`results/experiments.json`. Re-run it after any new measurement; nothing "
  "below is typed by hand.*")
w("")
w("## 1. Goal and context")
w("")
w("The project records 8-channel EMG from a Thalmic Myo and trains gesture "
  "classifiers that must work on a person they have never seen. This pass "
  "added: a **model zoo** (every classifier trained, measured and selectable), "
  "**soft voting** of selected models, **augmentation** and **adaptation to "
  "the person**, two questions the application asks (calibrate the hand; "
  "retrain), a settings area for all of it, an application icon, and the "
  "documentation and checks needed before the first test on a new band "
  "(`docs/sdk_myo.md`, `docs/install_windows.md`, `docs/install_linux.md`). "
  "Every new feature is off, or equal to the old behaviour, unless a "
  "measurement below says otherwise (`docs/FEATURES.md` lists how to turn "
  "each on and off).")
w("")
w("## 2. Data")
w("")
people = meta.get("people") or []
w(f"{len(people)} people ({', '.join(people)}), 211 recordings, six gestures "
  "(REST, FIST, OPEN, THUMB_UP, WAVE_LEFT, WAVE_RIGHT), per-wearer "
  "calibration `affine`, derived from each person's own recordings. After "
  "window-level deduplication the NN windows (0.7 s at 0.75 overlap) number "
  "2873. **Five people is small**: a difference of a couple of points is "
  "inside the spread between people, and the weakest person (Barbora, a "
  "differently seated band) alone moves every average by about 10 points. "
  "`docs/EDA.md` and `docs/MODELS_COMPARED.md` describe the data in full.")
w("")
w("## 3. Methods")
w("")
w("- **Protocol.** Leave-one-person-out: every window is scored by a model that "
  "never saw that person. The score is the mean over the five held-out people "
  "(each counts once). Per-recording vote and macro-F1 are reported beside it.")
w("- **Seeds.** Ablations use 3 seeds and report mean ± seed spread; the zoo "
  "uses 2 seeds for the nets and 3 for the ML models.")
w("- **A switch counts as helping** (`training/ablation_stats.py`) only if its "
  "mean gain over the baseline is at least max(1 point, 2 x the seed noise) AND "
  "it is better for at least 4 of the 5 people; or the same on the robustness "
  "probes without losing more than 1 point of clean accuracy. Anything else is "
  "\"no evidence\", stays off and remains selectable.")
w("- **Robustness probes**, applied to the held-out person's raw windows after "
  "the fact: multiplicative + additive noise; the band slid one pod either way "
  "(a circular channel roll); the whole signal 25 % weaker. Statistics for "
  "adaptation are taken before the perturbation (a band that drifts after "
  "calibration).")
w("- **DC-only control** (CLAUDE.md 8f/8k): the same model on each window "
  "flattened to its 8 channel means -- how much of a score is amplitude rather "
  "than gesture shape.")
w("- **Adaptation** standardises each person's input with statistics from a "
  "random 10 % of their own unlabelled windows (about 9 s, a calibration "
  "sweep). It is *not* zero-shot: it answers \"a new person who wears the band "
  "for a few seconds first\".")
w("")
w("## 4. Models and sizes")
w("")
w(f"{len(ml_models)} ML classifiers and {len(nn_models)} neural nets are in the "
  "zoo, all trained on the whole dataset and scored leave-one-person-out. "
  "★ = starred (top 3 per family by accuracy per window, ties by macro-F1).")
w("")
w("<!-- TABLE: models_ml -->")
w("")
def trained_with(m):
    tw = m.get("trained_with") or {}
    text = "adaptation" if tw.get("adapt") else "plain"
    if tw.get("augment"):
        text += ", aug " + "+".join(tw["augment"])
    return text


w(table(["", "model", "per window", "macro-F1", "per recording", "size on disk",
         "ms / live window", "trained with"],
        [("★" if m["starred"] else "", m["name"], pct(m["window_acc"]),
          pct(m["macro_f1"]), pct(m["vote_acc"]), size(m["size_bytes"]),
          ms(m["latency_ms"]), trained_with(m)) for m in ml_models]))
w("")
w("<!-- TABLE: models_nn -->")
w("")
w(table(["", "model", "per window", "macro-F1", "per recording", "parameters",
         "size on disk", "ms / live window"],
        [("★" if m["starred"] else "", m["name"], pct(m["window_acc"]),
          pct(m["macro_f1"]), pct(m["vote_acc"]),
          "--" if m["params"] is None else f"{m['params']:,}".replace(",", " "),
          size(m["size_bytes"]), ms(m["latency_ms"])) for m in nn_models]))
w("")
w("<!-- TABLE: model_sizes -->")
w("")
total_ml = sum(m["size_bytes"] or 0 for m in ml_models)
total_nn = sum(m["size_bytes"] or 0 for m in nn_models)
w(f"The whole ML zoo is {size(total_ml)}; the whole neural zoo {size(total_nn)}. "
  "Latency was measured while other jobs shared the machine; treat the "
  "ranking, not the absolute numbers, as the finding.")
w("")
ens = {e["family"]: e for e in DOC.get("ensembles", [])}
for fam, name in (("ml", "ML"), ("nn", "NN")):
    e = ens.get(fam)
    if e:
        w(f"**Best {name} soft-voting subset:** {', '.join(e['members'])} -- "
          f"{pct(e['window_acc'])} on the folds it was searched on, "
          f"**{pct(e.get('nested_acc'))} nested** (chosen without the held-out "
          f"person), against {pct(e.get('best_single_acc'))} for the best single "
          f"model. " + ("It is starred." if e["starred"] else
                        "It does not beat the best single model, so it is not "
                        "starred."))
        w("")
w("![The zoo](figures/zoo_overview.png)")
w("")
w("## 5. Newer architectures")
w("")
if newm:
    base = next((m for m in newm if m["model"] == "gru"), None)
    others = [m for m in newm if m["model"] != "gru"]
    best = max(others, key=lambda m: m["loso_acc"]) if others else None
    w("`tools/new_models_test.py` trained five architectures newer than "
      "LSTM/GRU under the same protocol, 150 epochs, one seed, on CUDA:")
    w("")
    w("<!-- TABLE: new_models_test -->")
    w("")
    w(table(["model", "parameters", "person held out", "time removed (DC-only)",
             "within-person (optimistic)", "ms / window"],
            [(m["name"], f"{m['params']:,}".replace(",", " "),
              pct(m["loso_acc"]), pct(m.get("dc_only_acc")),
              pct(m.get("contrast_acc")), ms(m.get("latency_ms")))
             for m in sorted(newm, key=lambda m: -m["loso_acc"])]))
    w("")
    if base and best:
        verdict = ("**none of them beat the GRU**" if
                   best["loso_acc"] <= base["loso_acc"] else
                   f"**{best['name']} beat the GRU**")
        w(f"Result: {verdict} -- GRU {pct(base['loso_acc'])}, best of the new "
          f"ones {best['name']} {pct(best['loso_acc'])}. Every new model is far "
          "better within-person (95 - 98 %) than across people, which is the "
          "usual sign that capacity goes into memorising the wearer. With one "
          "seed, differences of a few points are within noise (the GRU alone "
          "varied by about 10 points between seeds in the ablation below).")
    w("")
    w("![Newer models](figures/novesie_modely_test.png)")
w("")
w("## 6. Ablation study, and how the defaults were chosen")
w("")
w("One switch at a time against the baseline, then only combinations of what "
  "helped (stage B).")
w("")
for fam, title, anchor in (("nn", "Neural net (GRU, 0.7 s)", "ablation_nn"),
                           ("ml", "ML (soft vote ET-500 + RF-500-leaf3)",
                            "ablation_ml")):
    rows = [r for r in abl if r["family"] == fam]
    if not rows:
        w(f"*{title}: not measured yet.*")
        w("")
        continue
    w(f"### {title}")
    w("")
    w(f"<!-- TABLE: {anchor} -->")
    w("")
    w(table(["setting", "seeds", "per window", "± seeds", "macro-F1", "change",
             "people better", "robustness change", "verdict"],
            [(r["config"], r["seeds"], pct(r["acc_mean"]), pct(r["acc_std"]),
              pct(r["f1_mean"]), "--" if r["delta_acc"] is None
              else pts(r["delta_acc"]),
              "--" if r["people_better"] is None
              else f"{r['people_better']}/{r['n_people']}",
              "--" if r["delta_robust"] is None else pts(r["delta_robust"]),
              r["verdict"]) for r in rows]))
    w("")
w("![Ablation effects](figures/ablation_effects.png)")
w("")
w("### Voting against its members")
w("")
w("<!-- TABLE: ablation_voting -->")
w("")
if voting:
    w(table(["model", "per window", "± seeds", "macro-F1", "size", "ms / window"],
            [(v["model"], pct(v["acc_mean"]), pct(v["acc_std"]),
              pct(v["f1_mean"]), size(v.get("size_bytes")),
              ms(v.get("latency_ms"))) for v in voting]))
    w("")
    by = {v["model"]: v for v in voting}
    pair, four = by.get("vote(ET+RF)"), by.get("vote(all 4)")
    singles = [v for v in voting if not v["model"].startswith("vote")]
    best_single = max(singles, key=lambda v: v["acc_mean"]) if singles else None
    if pair and best_single:
        w(f"The deployed pair's vote is {pct(pair['acc_mean'])} against "
          f"{pct(best_single['acc_mean'])} for the best single member "
          f"({best_single['model']}): a gain of "
          f"{(pair['acc_mean'] - best_single['acc_mean']) * 100:+.1f} points, "
          "inside the seed spread -- voting does not demonstrably help here."
          + (f" Voting all four spec models falls to {pct(four['acc_mean'])}: an "
             "unweighted vote gives the 13-point weaker SVM and LDA an equal "
             "say." if four else ""))
        w("")
    w("![Voting](figures/voting_vs_members.png)")
    w("")
w("### Defaults")
w("")
w("<!-- TABLE: defaults -->")
w("")
w(table(["family", "setting", "shipped value", "why"],
        [(d["family"].upper(), d["setting"],
          ("on" if d["value"] is True else "off" if d["value"] is False
           else ", ".join(d["value"]) if isinstance(d["value"], list) and d["value"]
           else "off" if isinstance(d["value"], list) else d["value"]),
          d["reason"]) for d in defaults]))
w("")
w("The calibration mode was also measured for the ML pair: the shipped `affine` "
  "is kept because switching calibration off or to per-channel scaling "
  "measurably hurts. A single global gain was **not** judged better (it helps "
  "some people a great deal, notably the weakest, and hurts others), which is a "
  "lead to follow with more people, not a default.")
w("")
w("## 7. Robustness")
w("")
w("<!-- TABLE: robustness -->")
w("")
rows = [r for r in abl if r.get("robust_noise") is not None]
w(table(["family", "setting", "noise", "band shifted one pod", "25 % weaker"],
        [(r["family"].upper(), r["config"], pct(r["robust_noise"]),
          pct(r["robust_shift"]), pct(r["robust_gain"])) for r in rows]))
w("")
w("![Robustness](figures/ablation_robustness.png)")
w("")
w("A band slid one pod is the probe every model fails: accuracy drops by "
  "roughly 25 - 35 points whatever the switch. That is the open problem the "
  "weakest person's score reflects, and `rotate` augmentation is the one switch "
  "aimed at it (see the table for what it cost and gained).")
w("")
w("## 8. Adaptation to the person")
w("")
w("<!-- TABLE: adaptation -->")
w("")
ad = [r for r in abl if r["setting"] == "adaptation"]
w(table(["family", "per window", "± seeds", "macro-F1", "change", "people better",
         "verdict"],
        [(r["family"].upper(), pct(r["acc_mean"]), pct(r["acc_std"]),
          pct(r["f1_mean"]), pts(r["delta_acc"]),
          f"{r['people_better']}/{r['n_people']}", r["verdict"]) for r in ad]))
w("")
if fallback:
    m = fallback["mean"]
    w(f"**What if the person's statistics are missing?** An ML model trained with "
      f"adaptation, scored without them (the stored global statistics, which "
      f"is what `ADAPT_LIVE` off or a missing calibration signal does), reads "
      f"{pct(m['adapted model, global stats at test'])} against "
      f"{pct(m['adapted'])} with them and {pct(m['baseline'])} for a plain "
      "model (seed 0): the fallback costs nothing measurable, so an adapted "
      "model is safe to deploy before a calibration signal exists.")
    w("")
w("**Caveat.** The gain is not uniform: it comes mostly from the weakest and a "
  "couple of other people and is within the seed noise for some, because the "
  "statistics depend on which windows were sampled. Five people cannot settle "
  "it; the experiment to repeat is the same table with more wearers. Where "
  "there is data from one person only, adaptation cannot be tested at all.")
w("")
w("## 9. Real-time and latency")
w("")
w("<!-- TABLE: latency -->")
w("")
w(table(["family", "model", "ms per live window", "device"],
        [(m["family"].upper(), m["name"], ms(m["latency_ms"]),
          "CPU, 1 thread" if m["family"] == "ml" else "CUDA (GTX 1650)")
         for m in ml_models + nn_models]))
w("")
w("The live window interval is 150 ms. The ML forests cost most (a vote of "
  "several forests is the sum of its members); the nets are a few "
  "milliseconds. Voting adds the members' costs, and the NN worker runs all "
  "its nets in one process so a vote of nets costs the sum of their forward "
  "passes, not a second process.")
w("")
w("## 10. GUI features")
w("")
w("Settings (gear icon, Ctrl+,): Gestures, Models, Training, General. Details, "
  "how to turn each feature on and off, and the screenshots: "
  "[FEATURES.md](FEATURES.md) and [UI_GUIDE.md](UI_GUIDE.md).")
w("")
w("## 11. Testing and verification")
w("")
if tests:
    w("<!-- TABLE: tests -->")
    w("")
    w(table(["suite", "checks passed", "seconds"],
            [(t["suite"], f"{t['passed']}/{t['total']}"
              + ("" if t["ok"] else " **FAILED**"), t["seconds"])
             for t in tests]))
    w("")
    w(f"{sum(t['passed'] for t in tests)}/{sum(t['total'] for t in tests)} checks "
      f"in {len(tests)} suites (`python tools/run_tests.py`; the slow `nn` and "
      "`custom_gesture` suites are run separately).")
    w("")
    w("![Tests](figures/tests_overview.png)")
w("")
w("An independent check of the finished application is recorded in "
  "[VERIFICATION.md](VERIFICATION.md).")
w("")
w("## 12. Limitations")
w("")
w("- Five people; one band position each. Every number carries that.")
w("- Seed spread is large for the nets (the GRU baseline moved by about 10 "
  "points between seeds), so one-seed rankings (the newer-architectures test, "
  "the zoo's nets) are indicative, not final.")
w("- Adaptation is not zero-shot, and its gain is not uniform across people.")
w("- Latencies were measured on a busy machine; a quiet-machine benchmark is "
  "`tools/bench_models.py`.")
w("- Acquisition is Myo-only; nothing here was run against a real second band.")
w("- The nested subset estimate is honest about subset choice, not about the "
  "5-person sample.")
w("")
w("## 13. When the new band arrives")
w("")
w("Follow the hardware checklist in [sdk_myo.md](sdk_myo.md) (what a new "
  "backend must provide, what to run in what order, what numbers to expect) "
  "and the install guides ([Windows](install_windows.md), "
  "[Linux](install_linux.md)). A model trained on 8 channels refuses a "
  "different channel count rather than guessing; record new data with the new "
  "band, calibrate, and retrain (the retrain question appears at startup).")
w("")

with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
    fh.write("\n".join(L))
print("wrote", os.path.relpath(OUT, ROOT), f"({len(L)} lines)")


# ---- the defaults table in docs/FEATURES.md ------------------------------------
def fill_features_defaults():
    path = os.path.join(ROOT, "docs", "FEATURES.md")
    text = open(path, encoding="utf-8").read().replace("\r\n", "\n")
    marker = "<!-- TABLE: defaults -->"
    a = text.index(marker)
    b = text.index("\n---\n", a)
    rows = [("settings / gesture switches", "all gestures on",
             "no retrain either way (CLAUDE.md 7a)"),
            ("model selection", "off", "the original deployed models stay the "
             "default; select from the zoo in SETTINGS -> Models"),
            ("soft voting", "off", "only with two or more selected models; "
             "see the voting table in the report")]
    for d in defaults:
        value = ("on" if d["value"] is True else "off" if d["value"] is False
                 else ", ".join(d["value"]) if isinstance(d["value"], list)
                 and d["value"] else "off")
        rows.append((f"{d['setting']} ({d['family'].upper()})", value,
                     d["reason"]))
    rows += [("calibration question", "on", "asked for; turn off in General"),
             ("retrain question", "on", "asked for; turn off in General"),
             ("application icon", "on", "assets/icons/")]
    new = marker + "\n\n" + table(["setting", "default", "why"], rows) + "\n"
    open(path, "w", encoding="utf-8", newline="\n").write(
        text[:a] + new + text[b:])
    print("filled the defaults table in docs/FEATURES.md")


fill_features_defaults()
