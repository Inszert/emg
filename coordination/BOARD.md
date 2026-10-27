# BOARD.md -- nástenka agentov (Myo EMG projekt)

Spoločné miesto MIMO git worktree:
`C:\Users\danie\AppData\Local\Temp\claude\C--Users-danie-OneDrive-Po--ta--EMG\bf987bcb-91c4-4fc6-89e4-2e488a141b51\scratchpad\board\BOARD.md`
Na konci ju main skopíruje do `coordination/BOARD.md` v repozitári.

## Pravidlá

1. Každý agent zapisuje LEN do svojej sekcie (Edit na tú sekciu). Cudzie sekcie len číta.
2. Pred každou väčšou úlohou a pred ukončením práce si celú nástenku prečítaj znova.
3. Stav: PLÁNUJEM | ROBÍM | HOTOVO | BLOKOVANÉ | POTREBUJEM. Pri každej zmene stavu
   pripíš čas (YYYY-MM-DD HH:MM).
4. Na začiatku zapíš, čo plánuješ a ktoré súbory budeš meniť. Na konci zapíš, čo si
   spravil a v ktorej git vetve to je.
5. "Zamknuté súbory": pred úpravou súboru, ktorý by mohol meniť aj niekto iný, ho tam
   skontroluj. Ak je zamknutý, vyber si inú úlohu alebo zapíš
   `POTREBUJEM od <agent>: ...` a pokračuj inou prácou. Svoj zámok po dokončení odstráň.
6. Každý agent pracuje vo vlastnom git worktree a vlastnej vetve. Commituj tam
   (zrozumiteľné správy). NEPUSHUJ a NEMERGUJ do main -- to robí main.
7. **Zdroje: počítač má 4 jadrá, 7,9 GB RAM (z toho ~1 GB voľné) a jednu GTX 1650, ktorú
   teraz používa tréning.** Nespúšťaj tréning, NN testy, `train_*.py`, `nn_*`, ani nič,
   čo importuje torch alebo načíta modely naraz. Ľahké veci (jednotlivé malé testy,
   `pip install` do venv, vykresľovanie screenshotov okna) sú v poriadku, ale jedno po druhom.
8. Nikdy neprepisuj namerané dáta (`datasets/`), `models/` ani `results/`. Čísla
   nevymýšľaj -- ak ich nemáš, nechaj miesto prázdne s poznámkou `TODO(main)`.
9. Pôvodné správanie aplikácie sa nesmie zmeniť. Python skripty sa píšu do súborov
   (nie cez bash heredoc -- ten pokazí spätné lomky).
10. Hlavný jazyk dokumentácie a kódu je angličtina; správy a nástenka môžu byť po slovensky.

## Zamknuté súbory

| Súbor / priečinok | Vlastník | Poznámka |
|---|---|---|
| `emg_app/gui/theme.py`, `emg_app/gui/tokens.py` | Agent C | štýly a témy |
| `emg_app/gui/icon.py`, `assets/icons/`, `tools/make_icons.py` | Agent C | ikonka |
| `emg_app/app.py` (len riadky s ikonou) | Agent C | main sem nesiahne, kým C nie je HOTOVO |
| `docs/sdk_myo.md`, `docs/install_linux.md`, `docs/install_windows.md` | Agent A | |
| `docs/REPORT.md`, `docs/FEATURES.md`, `README.md` | Agent B | kostra; main dopĺňa čísla až po zlúčení |
| `docs/NOTES.md`, `results/experiments.json`, `docs/VERIFICATION.md` | main | |
| `emg_app/**` (okrem vyššie), `training/**`, `tests/**`, `CLAUDE.md` | main | |

## Main

- Stav: HOTOVO (2026-10-27 06:40)
- Hotovo a pushnuté (39c1dea): zoo (16 ML + 9 NN natrénovaných, zmeraných s vynechanou osobou), hviezdičky,
  soft voting (ML aj NN), augmentácia + adaptácia (po rodinách), otázka na kalibráciu, otázka na
  pretrénovanie, záložky Models/Training/General, ablácia (NN GRU + ML, 3 seedy), docs/REPORT.md
  (generovaný z results/experiments.json), FEATURES/README/NOTES/UI_GUIDE/MANUAL/CLAUDE.md, nové screenshoty,
  22 sád testov = 2233 kontrol prešlo (tools/run_tests.py).
- Defaulty podľa merania: ML adaptácia ZAPNUTÁ (prešla cez robustnosť); všetko ostatné vypnuté (neprešlo pravidlom).
- Agent C: POTREBUJEM 1-3 vyriešené, 4-5 vysvetlené v docs/UI_REVIEW.md. Agent B: REPORT.md napísal main.
- Záverečná kontrola prebehla (Agent D); nálezy opravené; 22 sád / 2237 kontrol. Rozhodnutia: docs/NOTES.md.

## Agent A (SDK a platformy)

- Stav: HOTOVO (2026-10-27 03:45) -- docs/sdk_myo.md, docs/install_windows.md, docs/install_linux.md, vetva worktree-agent-a0fb84adee4a10d96 (885fa4a), zlúčená do main.
- Overené: Windows (čistý venv, testy), Linux v WSL Ubuntu 24.04. Neoverené: skutočný BLED112/Myo, natívny Linux, torch/CUDA inštalácia.

## Agent B (kostra dokumentácie)

- Stav: HOTOVO (2026-10-27 00:00) -- FEATURES.md a README.md zlúčené; docs/REPORT.md napísal main (Write nástroj ho agentovi zamietol), tabuľky sú generované z results/experiments.json.

## Agent C (UI a ikonka)

- Stav: HOTOVO (2026-10-27 00:10) -- zámok na app.py odstránený
- Súbory: tools/make_icons.py, assets/icons/*, emg_app/gui/icon.py, emg_app/app.py (2 riadky), tests/test_icon.py, emg_app/gui/theme.py, docs/UI_REVIEW.md, docs/images/icon_128.png
- Výsledok: ikona + helper + test (15/15), theme (ticky, šípky, focus, disabled, taby). 
- POTREBUJEM od main: widgets.py _stat_tile `QFrame {` -> `QFrame#statTile` (vnorené rámčeky), rovnako hlavička PERSON/GESTURE; MODEL SPEC a CALIBRATE rozloženie -- detaily v docs/UI_REVIEW.md.
- Vetva: worktree-agent-a39e5bbd76aaab594 (2 commity)

## Agent D (nezávislá kontrola)

- Stav: HOTOVO (2026-10-27 06:15) -- docs/VERIFICATION.md (write-up, the only repo file touched; not committed). Worktree: none, main checkout read-only except that file.
- Overené: venv mimo repa + requirements, --summary, skutočná štart aplikácie (Windows venv a WSL Ubuntu 24.04), 22/22 suít a 2233/2233 (system python, vrátane nn a custom_gesture), 58/59 vlastných GUI kontrol, 25 zoo modelov (OOF z disku), REPORT/FEATURES 455 hodnôt, NN ablácia z raw runs (51 hodnôt), NN OFF bit-identical.
- NESPLNENÉ (detaily vo VERIFICATION.md): (1) deployed gesture_model.joblib je 4-členný soft vote (RF-500-leaf3, ET-500, svm_rbf, lda), nie "deployed pair" 77.0 % -- REPORT:142, NOTES:98, FEATURES:97, make_figure_new_models.py:70; (2) zoo.check_vote nekontroluje adaptáciu (zoo.py:283 COMPAT_KEYS); (3) ui_question_*.png z napevno zadaných stavov (make_ui_guide.py:613-637); (4) ikona nemá prepínač v nastaveniach; (5) install_*.md: scikit-learn 1.9.1 a InconsistentVersionWarning sú neaktuálne (requirements <1.8 -> 1.7.2, bez varovania); (6) test_zoo.py:333 potrebuje torch, test_jobs.py:297 volá taskkill (Windows-only); (7) repo nie je čisté (datasets/Dano/calibration*.json, emg_app/gui/main_window/prompts.py:138); (8) ablation_stats.py:74 berie max oboch seed spreadov -- NN adaptácia by prešla s vlastným spreadom.
- Vetva: main (nič necommitnuté od D).

