# Archived dataset — 2026-10-15, before the R1 re-record

325 recordings, 1 subject, 2 sessions, 6 labels, 7 free takes. Moved here
whole, not deleted: safety rules 4 and 5 say recordings are quarantined and
never erased, and that applies hardest to "delete everything".

The leading underscore is what makes it invisible — `discover_recordings()`
does `dirnames[:] = [d for d in dirnames if not d.startswith("_")]` at every
level of the walk, so nothing in here can re-enter the dataset by accident.

## Why it was retired

Every active gesture was **one single free take**: FIST 1 source, WAVE_LEFT 1,
OPEN 5, WAVE_RIGHT 6, THUMB_UP 9, REST 30 — 325 recordings but only 52
distinct sources, because 280 of them are clips tiling five long takes with
~50 % overlap. So the label was perfectly confounded with electrode state and
wall-clock block, and the model's 99.6 % was reading *which 50-second
recording it was looking at*. The DC-only control proved it: 8 channel means
scored 98.9 %, exactly what the full 50-feature vector scored. The other 42
features added +0.0 points.

It also had no onsets, no offsets and no transitions anywhere (every take was
~100 % contraction), and no relaxed holds for FIST/WAVE_LEFT/REST — which is
why a held thumbs-up flipped to REST live. `docs/EDA.md` has the full
analysis; CLAUDE.md §8f summarises it.

## What replaces it

Cued sessions via `START SESSION`, which nothing in this archive used — 0 of
325 recordings came from `SequenceController`. Five sittings with the armband
removed and refitted between them, gestures interleaved, `RECORD_TIME = 4.0`
so each rep captures onset -> settle -> relaxed hold -> release.

## It is still useful

Keep it. It is the baseline that shows what the confound looked like, and the
only data that can answer "did the re-record actually change anything". When
the new dataset is trained, compare its DC-only control against this one's
98.9 %.

## To restore

```powershell
Move-Item datasets\_archive_pre_r1_20261015\Dano datasets\Dano
```
