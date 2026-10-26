"""
Which models the zoo holds, and what each one is. No imports of anything heavy.

ML: the sixteen classifiers of the combination test (docs/COMB_TEST.md), all at
the winning feature setup (bandpass / `level` features of 1.0 s windows at 0.75
overlap), so any of them can be voted with any other. The entries are
spec-shaped (name, kind, scaler, params) -- `train_model._make_pipeline` builds
them, the same code the trainer uses.

NN: every architecture in `nn_arch.KINDS`, at 0.7 s windows (the NN setup from
the combination test). The nets can be voted with each other, not with the ML
models.
"""
from __future__ import annotations

ML_SETUP = {"window_s": 1.0, "overlap": 0.75, "representation": "bandpass",
            "freq_features": False, "feature_set": "level"}
NN_SETUP = {"window_s": 0.7, "overlap": 0.75}

ML_ROSTER = [
    ("LDA-shrinkage", "lda", "standard",
     {"solver": "lsqr", "shrinkage": "auto"}),
    ("LogReg-C0.1", "logreg", "standard", {"C": 0.1}),
    ("LogReg-C1", "logreg", "standard", {"C": 1.0}),
    ("SVM-rbf-C1", "svm_rbf", "standard", {"C": 1.0}),
    ("SVM-rbf-C3", "svm_rbf", "standard", {"C": 3.0}),
    ("SVM-rbf-C10", "svm_rbf", "standard", {"C": 10.0}),
    ("SVM-rbf-C30", "svm_rbf", "standard", {"C": 30.0}),
    ("RF-500", "random_forest", "none", {"n_estimators": 500}),
    ("RF-500-leaf3", "random_forest", "none",
     {"n_estimators": 500, "min_samples_leaf": 3}),
    ("ET-500", "extra_trees", "none", {"n_estimators": 500}),
    ("ET-500-leaf3", "extra_trees", "none",
     {"n_estimators": 500, "min_samples_leaf": 3}),
    ("HGB-lr0.05", "hist_gradient_boost", "none",
     {"learning_rate": 0.05, "max_iter": 300}),
    ("kNN-15", "knn", "standard", {"n_neighbors": 15}),
    ("kNN-31", "knn", "standard", {"n_neighbors": 31}),
    ("MLP-128", "mlp", "standard",
     {"hidden_layer_sizes": [128], "alpha": 0.001}),
    ("MLP-256-128", "mlp", "standard",
     {"hidden_layer_sizes": [256, 128], "alpha": 0.01}),
]

#  Display names for the nets (the id is nn-<kind>).
NN_TITLES = {
    "tcn": "NN-tcn", "gru": "NN-gru", "cnn_gru": "NN-cnn_gru",
    "lstm": "NN-lstm", "cnn": "NN-cnn", "circ_gru": "NN-circ_gru",
    "chan_transformer": "NN-chan_transformer", "inception": "NN-inception",
    "tsmixer": "NN-tsmixer",
}


def slug(text):
    out = "".join(c.lower() if c.isalnum() or c == "." else "-" for c in text)
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-")


def ml_id(name):
    return "ml-" + slug(name)


def nn_id(kind):
    return "nn-" + slug(kind.replace("_", "-"))
