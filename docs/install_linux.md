# Installing on Linux (Ubuntu / Debian)

> **Not verified on a real Linux machine with a Myo.** What was tested: an
> Ubuntu 24.04.5 LTS under **WSL2** (Python 3.12.3) on the documentation
> machine, which has **no USB, no Bluetooth dongle and no armband**. So the
> install, the offline tests and the GUI start-up are verified; **anything that
> touches the BLED112 dongle or the armband on Linux is not.** The section
> "What is verified and what is not" at the end lists it exactly. Steps are
> labelled **Verified (WSL)** or **Not verified**.

Date of the checks: 2026-10-26/27. Fresh venv, project copied into the WSL
home folder (the Windows drive is slow under WSL).

## 1. System packages

```bash
sudo apt-get update
sudo apt-get install -y python3-venv libxcb-cursor0
```

**Verified (WSL):** on a stock WSL Ubuntu 24.04, `python3 -m venv` failed with
`No module named 'ensurepip'` until `python3-venv` was installed. Checked as
already present and **not** needed to install: `libegl1`, `libgl1`,
`libxkbcommon0`, `libxkbcommon-x11-0`, `libdbus-1-3`, `libfontconfig1`,
`libglib2.0-0t64`, `libxcb-icccm4`, `libxcb-image0`, `libxcb-keysyms1`,
`libxcb-randr0`, `libxcb-render-util0`, `libxcb-shape0`, `libxcb-xinerama0`,
`libxcb-xkb1`, `libnss3`. On a minimal server image these may be missing; if
Qt complains about a library, `ldd` the platform plugin to find which:

```bash
ldd .venv/lib/python3*/site-packages/PySide6/Qt/plugins/platforms/libqxcb.so | grep "not found"
```

**Verified (WSL):** before installing `libxcb-cursor0` this printed
`libxcb-cursor.so.0 => not found`, and starting the app without
`QT_QPA_PLATFORM=offscreen` aborted with

```
qt.qpa.plugin: From 6.5.0, xcb-cursor0 or libxcb-cursor0 is needed to load the Qt xcb platform plugin.
This application failed to start because no Qt platform plugin could be initialized.
```

After `sudo apt-get install -y libxcb-cursor0` the window started (WSLg
display). That is the single package PySide6 6.12 needed beyond the stock
image there. (Debian package name is the same.)

## 2. Python environment

Needs Python 3.12 or newer in practice (the project is developed on 3.14;
**3.12.3 was verified here**, all wheels have manylinux builds).

```bash
cd /path/to/EMG
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Verified (WSL):** succeeded. Resolved:

```
numpy 2.5.4  pandas 3.1.0  scipy 1.18.1  scikit-learn 1.9.1  joblib 1.6.0
pyarrow 26.0.0  pyserial 3.5  PySide6 6.12.0  pyqtgraph 0.14.0  shiboken6 6.12.0
```

(PySide6 pulls the large Addons/WebEngine wheels; allow time and about 2-3 GB.)
The saved model was pickled with scikit-learn 1.7.2, so a clean install prints
`InconsistentVersionWarning`; see `install_windows.md` "Known issues".

## 3. Running the app and the tests

```bash
export PYTHONPATH=.                     # from the project root
python emg_dataset_tool.py --summary    # no GUI
python tests/test_datalayer.py
python tests/test_device.py
python tests/test_no_torch.py
python tests/test_gui.py                # offscreen GUI with a simulated armband
python emg_dataset_tool.py              # the app
```

**Verified (WSL):** `--summary` exit 0 (211 recordings); `test_datalayer`
150/150; `test_device` 104/104; `test_no_torch` 11/11; `test_gui` 109/109 (the
tests set `QT_QPA_PLATFORM=offscreen` themselves). The app started under WSLg
with `libxcb-cursor0` installed and printed

```
[startup] building the interface ...
[myo] Myo error: Myo dongle not found!
[startup] window up after 4.8 s
```

and kept running until stopped by `timeout 25`.

**Headless / no display** (servers, CI): `QT_QPA_PLATFORM=offscreen python
emg_dataset_tool.py` also works for the tests; for the interactive app you need
an X11 or Wayland session. Over SSH use `ssh -X` or a VNC/xvfb session
(`xvfb-run`, not tried).

Only run the light tests above on a small machine; the trainers
(`training/train_*.py`) are CPU- or GPU-heavy.

## 4. Serial port, permissions, udev (BLED112)

**Not verified (no dongle on this machine).** What the code does, read from
`emg_app/myo/myo_raw.py`:

* `MyoRaw.detect_tty()` uses `serial.tools.list_ports.comports()` and matches
  the regex `PID=2458:0*1` against each port's hardware-id string. That is
  portable pyserial behaviour: on Linux the hwid is of the form
  `USB VID:PID=2458:0001 SER=... LOCATION=...`, so the same expression
  applies. **Verified (WSL):** `comports()` returns `[]` and `MyoRaw()` raises
  `ValueError: Myo dongle not found!` as designed; the matching expression
  itself was checked on a sample string (`docs/sdk_myo.md`).
* The port is opened with `serial.Serial(port=tty, baudrate=9600, dsrdtr=1)`,
  nothing Windows-specific.
* To override auto-detection: `MYO_PORT = "/dev/ttyACM0"` in
  `emg_app/config.py`.

Expected on a real machine (standard Linux behaviour, **not tested here**):
the BLED112 appears as a CDC-ACM device, `/dev/ttyACM0`, owned by `root:dialout`
(Ubuntu/Debian). Check with `ls -l /dev/ttyACM*` after plugging it in, and
`dmesg | tail`.

1. Add yourself to the group, then log out and in:

   ```bash
   sudo usermod -aG dialout "$USER"
   ```

2. Optional udev rule so the port always has the right group and gets a stable
   name, and so ModemManager does not probe the dongle (a common cause of
   "port busy" on desktop Ubuntu). Save as
   `/etc/udev/rules.d/99-myo-bled112.rules`:

   ```
   # Bluegiga BLED112 BLE dongle used with the Myo armband
   SUBSYSTEM=="tty", ATTRS{idVendor}=="2458", ATTRS{idProduct}=="0001", MODE="0660", GROUP="dialout", SYMLINK+="bled112", ENV{ID_MM_DEVICE_IGNORE}="1"
   ```

   then `sudo udevadm control --reload && sudo udevadm trigger` and replug.
   **Verified (WSL):** `udevadm verify` (systemd 255) reports "1 udev rules files
   have been checked. Success: 1 Fail: 0" -- syntax only; the rule has never
   matched a real device. If you use the `/dev/bled112` symlink, set
   `MYO_PORT = "/dev/bled112"`.
3. If ModemManager is installed (it is on desktop Ubuntu; it is on this WSL
   image too) and the port still gets grabbed, the `ID_MM_DEVICE_IGNORE` line
   above is the fix; as a blunt test, `sudo systemctl stop ModemManager`.
4. **Bluetooth stack:** the BLED112 does the BLE itself over serial (BGAPI), so
   BlueZ / the machine's own Bluetooth is not used and does not need to be
   enabled.
5. **WSL note:** WSL2 has no USB access by default. To use the dongle there you
   need `usbipd-win` to attach it (`usbipd attach --wsl`), then it shows up as
   `/dev/ttyACM0`. Not tried; native Linux is the safer route.

## 5. Is anything Windows-only? (honest answer)

`CLAUDE.md` section 12 says "Windows only, in practice" because nothing else
was ever tried, not because a Windows API is required on the **recording path**.
Reading the code:

* **Recording path (`myo_raw.py`, `acquisition.py`, data layer): no
  Windows-specific code found.** The one thing that is performance-sensitive
  on Windows (re-assigning `ser.timeout` per byte reconfigures the COM port,
  `sdk_myo.md` rule 1) is avoided on every platform by calling `run()` without
  a timeout. Whether Linux has an equivalent hazard is **unmeasured**; the
  rule applies anyway.
* **Background training jobs** (`emg_app/jobs.py`, `training/train_job.py`)
  have an `os.name == "nt"` branch for Windows and a POSIX branch marked
  `# pragma: no cover` (never exercised by the tests, **not run here**):
  detach via `start_new_session=True`, liveness via `os.kill(pid, 0)`.
  One weakness visible in the code: on POSIX `kill_tree()` only does
  `proc.kill()` on the trainer, **not its joblib worker processes** (Windows
  uses `taskkill /T`), so cancelling a training job may leave workers running.
  Reported to main.
* **Fonts:** the UI asks for `Segoe UI` and `Consolas`
  (`emg_app/gui/theme.py`, `plot.py`, ...). On Linux Qt substitutes other fonts;
  layouts were tuned with Windows metrics, so spacing may differ slightly
  (the GUI test passed, but looks were not inspected).
* Paths use `os.path` throughout; `recording.py` normalises `\\` when parsing
  paths. The `--summary` and test runs above used Linux paths successfully.

## 6. Optional: CUDA PyTorch (neural net only)

Same command as Windows (from `requirements.txt`); the recorder never needs it:

```bash
python -m pip install torch==2.9.1+cu128 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**Not run** (no torch install was allowed here, and the WSL image has no
NVIDIA setup checked). Needs an NVIDIA driver on the host; under WSL2 the
driver is the Windows one. Whether a `+cu128` wheel exists for your Python
version on that index was not checked. `training/train_nn.py` refuses a
CPU-only torch rather than running slowly.

## 7. Common errors

| Symptom | Fix |
|---|---|
| `No module named 'ensurepip'` creating the venv | `sudo apt-get install python3-venv` |
| `Could not load the Qt platform plugin "xcb"` / `xcb-cursor0 ... is needed` | `sudo apt-get install libxcb-cursor0`; or run with `QT_QPA_PLATFORM=offscreen` for tests |
| `Failed to create wl_display` | harmless message when a Wayland socket is absent; Qt falls back to xcb |
| `Myo dongle not found!` | dongle not attached / permissions / udev; run the `comports()` check in `docs/sdk_myo.md` |
| `Permission denied: '/dev/ttyACM0'` | not in group `dialout` (log out and in after `usermod`) |
| port busy / dongle behaves oddly right after plugging | ModemManager probing it; use the udev rule above |
| `ModuleNotFoundError: emg_dataset_tool` in a test | `export PYTHONPATH=.` from the project root |

## What is verified and what is not

Verified (Ubuntu 24.04.5 on WSL2, Python 3.12.3, 2026-10-26/27):
python3-venv requirement; `pip install -r requirements.txt` (versions above);
the missing `libxcb-cursor0` failure and its fix; `--summary`; `test_datalayer`
150/150, `test_device` 104/104, `test_no_torch` 11/11, `test_gui` 109/109;
app start-up to "window up" with no dongle; `detect_tty` failing cleanly with
no dongle; udev rule syntax with `udevadm verify`.

**Not verified:** a real native Linux machine; any real BLED112 dongle (USB
enumeration as `ttyACM0`, group ownership, the udev rule matching, ModemManager
interaction, usbipd under WSL); any connection to a real armband from Linux
(scan, streaming rate, reconnects, `run()` throughput on Linux); Debian or
other Ubuntu versions; Wayland-only sessions; the POSIX branch of the training
job runner; CUDA/torch on Linux; font appearance.

Side effects on the test machine: `python3-venv` and `libxcb-cursor0` were
installed into the WSL Ubuntu-24.04 distro with apt, and the throwaway copy of
the project in `~/emgtest` there was left in place (safe to delete).
