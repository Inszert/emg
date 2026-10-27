# Installing on Windows

From a clean machine to a running app. Everything marked **Verified** was
really executed on 2026-10-26 on Windows 11 (10.0.26300) with CPython 3.14.0,
in a **fresh virtual environment created outside the repository**, with no
armband and no BLED112 dongle attached. Steps that need the dongle, the
armband or a GPU are marked as such and were *not* executed.

## 0. What you need

| Item | Notes |
|---|---|
| Windows 10/11, 64-bit | the project is developed and used on Windows 11 |
| Python **3.14** (64-bit) | `requirements.txt` says "verified on Windows 11 with CPython 3.14". Verified here with 3.14.0; other versions were **not** tried (every dependency has a cp314 wheel, so 3.12/3.13 are probably fine, but that is untested) |
| about 3 GB free disk | the PySide6 wheels are large (`PySide6_Addons` and `PySide6_WebEngine` come with `PySide6`) |
| Myo armband + Bluegiga **BLED112** USB dongle (USB VID:PID `2458:0001`) | only for recording and live prediction. Everything else (viewing, training on the existing data, running the tests) works without them |
| optional: NVIDIA GPU + driver | only for the neural net (section 6); the recorder and the scikit-learn model never need it |

## 1. Python

Install CPython 3.14 (64-bit) from python.org and tick **Add python.exe to
PATH**, or use the `py` launcher. Check:

```powershell
python --version
```

**Verified:** `Python 3.14.0`.

## 2. Get the project and create a virtual environment

```powershell
cd C:\path\to\EMG                 # the project root (contains emg_dataset_tool.py)
python -m venv .venv              # or put it anywhere, e.g. C:\venvs\emg
.\.venv\Scripts\Activate.ps1      # if PowerShell blocks scripts: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The project path may contain non-ASCII characters (the owner's checkout lives
under `...\Počítač\EMG`); the app forces a UTF-8 console for that reason.

**Verified:** the venv was created with `python -m venv <folder>` outside the
repo, and `pip install -r requirements.txt` succeeded on the first try
(exit code 0; on a slow connection the 131 MB `PySide6_WebEngine` wheel needed
several automatic resumes, which pip handles itself, so allow 10 minutes or
more on a slow link). Resolved versions that day:

```
Python 3.14.0 | pip 25.2
numpy 2.5.4  pandas 3.1.0  scipy 1.18.1  scikit-learn 1.7.2  joblib 1.6.0
pyarrow 26.0.0  pyserial 3.5  PySide6 6.12.0  (+Addons, Essentials, Pdf, WebEngine)
pyqtgraph 0.14.0  shiboken6 6.12.0
```

`requirements.txt` only gives lower bounds (`>=`), so a new install gets
**newer** versions than the ones the project was developed with
(see "Known issues found" below for the one consequence that matters:
`scikit-learn`).

## 3. The BLED112 dongle and the COM port

Not verified here (no dongle). What the code needs, and how to check it:

1. Plug the dongle in. Windows normally binds its built-in USB serial (CDC)
   driver, and a new entry such as **Bluegiga Bluetooth Low Energy** /
   *USB Serial Device (COMx)* appears under **Ports (COM & LPT)** in Device
   Manager. If it shows as an unknown device, install the Silicon Labs / Bluegiga
   driver for the BLED112 (the `2458:0001` hardware id identifies it).
2. The app finds it by USB id, so no port number is needed
   (`MYO_PORT = None` in `emg_app/config.py`). This check, from the project
   root with the venv active, lists what the code will see:

   ```powershell
   python -c "import re; from serial.tools.list_ports import comports; [print(p.device, p.hwid) for p in comports()]; print([p.device for p in comports() if re.search(r'PID=2458:0*1', p.hwid)] or 'BLED112 dongle: not found')"
   ```

   **Verified (without a dongle):** prints `BLED112 dongle: not found`. With the
   dongle plugged in it must print its port (for example `['COM3']`); the
   owner's machine uses `COM3`.
3. Only one program can hold the port. Close other Myo tools first. To force a
   port, set `MYO_PORT = "COM3"` in `emg_app/config.py`.
4. Wake the armband (flex, or shake it) before connecting; it stops advertising
   a minute or two after it is taken off.

## 4. Run the application

From the project root, venv active:

```powershell
python emg_dataset_tool.py --summary     # dataset statistics, no GUI, no hardware
python emg_dataset_tool.py               # the application
```

**Verified:** `--summary` exit code 0; output (abridged):

```
Dataset root: ...\datasets
Raw recordings: 211   Trimmed: 0

  Barbora
    session_01  (43)
      FIST           4
      ...
```

with five subjects (Barbora, Betka, Bruno, Dano, Miso).

**Verified:** starting the full app (headless, `QT_QPA_PLATFORM=offscreen`,
dongle absent) printed

```
Myo EMG Dataset Tool -- loading libraries ...
[startup] building the interface ...
[myo] Myo error: Myo dongle not found!
[startup] window up after 10.4 s (press Ctrl+C in this console, or close the window, to quit)
```

and stayed running for 25 s until it was stopped. "Myo dongle not found!"
is the expected message when no dongle is plugged in; the window keeps
retrying (`AUTO_RECONNECT`). Start-up takes a few seconds (10.4 s on the
documentation machine) because the model and Qt are loaded.

Command-line flags (`emg_app/app.py`): `--summary`; `--train` /
`--train-quick` start a retrain as soon as the window is up (this runs a
CPU-heavy sweep; do not use it on a busy machine).

First use with a band: `docs/sdk_myo.md` section 7 is the step-by-step
hardware checklist. A **new wearer must press CALIBRATE once** (six seconds)
before the classifier will predict for them.

## 5. Tests that are safe to run (no hardware, no GPU, no torch)

Run them from the project root. Tests set `QT_QPA_PLATFORM=offscreen`
themselves, so no window appears. Some need the project root on the path:

```powershell
$env:PYTHONPATH = "."
python tests\test_datalayer.py     # 150 checks
python tests\test_device.py        # 104 checks
python tests\test_gui.py           # offscreen GUI with a simulated armband
python tests\test_filters.py       # 52 checks
python tests\test_no_torch.py      # 11 checks, proves the app needs no torch
```

**Verified** in the fresh venv: `test_datalayer.py` 150/150 (needs
`PYTHONPATH=.`; without it: `ModuleNotFoundError: No module named
'emg_dataset_tool'`), `test_device.py` 104/104 (adds the path itself),
`test_gui.py` **109/109** in 37 s (the count in `tests/README.md`, 91, is out
of date), `test_filters.py` 52/52 in 8 s, `test_no_torch.py` 11/11 in 6 s.
Other no-hardware scripts (`test_review`, `test_markers`,
`test_label_on_graph`, `test_free_take`, `test_calibration`,
`test_class_mask`, `test_model_spec`, `test_trail`, `test_jobs`) exist and are
listed in `tests/README.md`; they were **not** run for this document.

`python tools/run_tests.py` runs every offline suite (`--fast` skips the two slow ones); it was run by the independent check, see docs/VERIFICATION.md.

Do **not** run on a busy or small machine: `test_predict.py` (trains a toy
model and runs a real retrain), `test_custom_gesture.py` (about 5 min, trains),
`test_nn.py`, `test_torch.py` (GPU), and the `test_live*.py` scripts, which need
the armband (see `sdk_myo.md`).

## 6. Optional: PyTorch with CUDA (neural net only)

The recorder and `LivePredictor` **never import torch**; only the separate
neural-net trainer (`training/train_nn.py`), its live worker
(`training/nn_serve.py`) and `tests/test_nn.py` need it. It is deliberately
not in `requirements.txt` because pip's default index serves a CPU-only build,
which the trainer refuses. Install it explicitly (command from
`requirements.txt`):

```powershell
python -m pip install torch==2.9.1+cu128 --index-url https://download.pytorch.org/whl/cu128
```

**Not run here** (the machine's GPU was busy and the download is about 2-3 GB).
The project owner's setup that this version was validated on: GeForce GTX
1650 (4 GB), CUDA 12.8, torch 2.9.1+cu128. Whether `torch==2.9.1+cu128` has a
wheel for Python 3.14 on the index was **not checked**; the owner's working
environment is 3.14, so it did at the time. Check afterwards with:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

scikit-learn has no GPU backend: RETRAIN is CPU work whatever you install.

## 7. Common errors

| Symptom | Cause / fix |
|---|---|
| `Myo dongle not found!` | dongle not plugged in, no driver (Device Manager shows an unknown device), or `MYO_PORT` points elsewhere. Run the check in section 3 |
| `could not open port 'COM3': PermissionError(13, 'Access is denied.' ...)` (the usual pyserial wording; not reproduced here) | another program (a second copy of the app, `legacy/test_myo.py`, a serial terminal) holds the port |
| `Armband did not answer in 20 s -- it is probably asleep` | wake the band (flex, shake) and keep wearing it; it stops advertising soon after being taken off |
| `A device attached to the system is not functioning` | the port was being reconfigured in a storm (someone passed a timeout to `myo.run()`, see `sdk_myo.md` rule 1) or the dongle is half-dead: unplug and replug it |
| `PySide6 and pyqtgraph are required for the GUI.` | the venv is not active, or `pip install -r requirements.txt` did not finish |
| `ModuleNotFoundError: No module named 'emg_dataset_tool'` when running a script from `tests\` | set `$env:PYTHONPATH = "."` from the project root |
| `UnicodeEncodeError` printing a path | use the project's entry point (it forces UTF-8); in your own scripts run `chcp 65001` or `python -X utf8` |
| PowerShell: "running scripts is disabled" when activating the venv | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or skip activation and call `.\.venv\Scripts\python.exe` directly |
| `InconsistentVersionWarning: Trying to unpickle estimator ... from version 1.7.2 when using version 1.9.1` | scikit-learn is newer than the pin: `python -m pip install "scikit-learn>=1.3,<1.8"` |
| `torch` not found / NN panel says what is missing | expected without torch; install per section 6 or ignore |
| the app refuses to predict for a wearer: "this model was trained on 'affine' per-wearer calibration, and <person> has none" | press CALIBRATE (or set the calibration mode to `off` and retrain) |

## Known issues found while writing this

* **scikit-learn version.** The saved models were pickled with 1.7.2, and `requirements.txt` now pins `scikit-learn>=1.3,<1.8`, so a clean install resolves 1.7.2 and loads them with no `InconsistentVersionWarning` (verified by the independent check, docs/VERIFICATION.md). Do not upgrade scikit-learn past 1.7.x without retraining.
* `PySide6` brings `PySide6_Addons` and `PySide6_WebEngine` (about 130 MB
  alone) although the app does not use WebEngine. `PySide6-Essentials` would be
  enough for the app but was **not** tried.
