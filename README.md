# emg

Labeled 8-channel EMG gesture recording tool for the Thalmic Myo armband, plus
the training and validation pipeline that goes with it.

The deliverable is the **recording tool**, not a model: the point is a clean,
honestly-labeled dataset with subject-independent validation.

## Run it

```powershell
python -m pip install -r requirements.txt     # once
python emg_dataset_tool.py                    # the application
python emg_dataset_tool.py --summary          # dataset stats, no GUI
python train_model.py                         # scikit-learn sweep
python train_torch.py                         # CUDA 1D-CNN / LSTM / CNN-LSTM
```

Needs a Myo armband and a Bluegiga BLED112 dongle. Windows, in practice.

## Where to read

| File | What it covers |
|---|---|
| [`GUIDE.md`](GUIDE.md) | user manual: every file, every setting, ready-made presets |
| [`CLAUDE.md`](CLAUDE.md) | orientation and the invariants — read before changing anything |
| [`problems.md`](problems.md) | every failure hit, the fix tried, and whether it actually worked |
| [`recommendation.md`](recommendation.md) | what to collect and train next, measured against the data on disk |
| [`docs/DATASET.md`](docs/DATASET.md) | on-disk format, filenames, CSV/NPZ schema |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | threads, buffers, state machine, data flow |
| [`docs/ML_PIPELINE.md`](docs/ML_PIPELINE.md) | recordings to a validated model |
| [`tests/README.md`](tests/README.md) | twelve check scripts; three need the armband |

## State

One subject, one session, FIST/REST only. Six-gesture accuracy was ~43 % on an
earlier set and the dataset is the bottleneck, not the model — see
[`problems.md`](problems.md) §4 before trusting any number in here.
