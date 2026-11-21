"""Reproducible Dry Bean experiments based on an AI Fundamentals coursework notebook."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import urllib.request
import zipfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATA_URL = "https://archive.ics.uci.edu/static/public/602/dry%2Bbean%2Bdataset.zip"
DATA_SHA256 = "0a64eff5be87f48c3dbbfc0a12a56c5d5b5167ef8e61cd45d69b3e7c7130c06f"
FEATURES = ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "AspectRation",
            "Eccentricity", "ConvexArea", "EquivDiameter", "Extent", "Solidity",
            "roundness", "Compactness", "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4"]
CLASSES = ["BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"]
ARCHITECTURES = {"MLP A": [64, 32], "MLP B": [16], "MLP C": [128, 64, 32]}


def clean_data(df):
    """Validate the source schema, then remove exact feature+label duplicates."""
    if set(df.columns) != set(FEATURES + ["Class"]):
        raise ValueError("Expected the 16 UCI feature columns and Class")
    df = df[FEATURES + ["Class"]].copy()
    df["Class"] = df["Class"].map(lambda v: v.decode() if isinstance(v, bytes) else v)
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(df[FEATURES].to_numpy()).all() or not df.Class.isin(CLASSES).all():
        raise ValueError("Data contains missing/non-finite features or an unknown class")
    if (df.groupby(FEATURES, dropna=False).Class.nunique() > 1).any():
        raise ValueError("Identical feature vectors have conflicting labels")
    return df.drop_duplicates().copy()


def load_data(archive):
    archive = Path(archive)
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix(".download")
        try:
            with urllib.request.urlopen(DATA_URL, timeout=180) as source, temporary.open("wb") as target:
                import shutil
                shutil.copyfileobj(source, target)
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    if checksum != DATA_SHA256:
        raise ValueError("Dataset archive checksum differs from the recorded UCI source")
    with zipfile.ZipFile(archive) as source:
        member = next(n for n in source.namelist() if n.endswith(".arff"))
        with source.open(member) as stream:
            import io
            records, _ = arff.loadarff(io.TextIOWrapper(stream))
    raw = pd.DataFrame(records)
    clean = clean_data(raw)
    return clean, {"url": DATA_URL, "archive_sha256": checksum,
                   "raw_rows": len(raw), "duplicates_removed": len(raw) - len(clean), "rows": len(clean)}


def split_data(df, seed=42):
    """Preserve the original coursework's approximate 70/15/15 stratified split."""
    trainval, test = train_test_split(df.index.to_numpy(), test_size=.15, random_state=seed, stratify=df.Class)
    train, val = train_test_split(trainval, test_size=.1765, random_state=seed, stratify=df.loc[trainval, "Class"])
    return {"train": train, "validation": val, "test": test}


def scores(y, pred):
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0))}


def build_mlp(hidden, seed=42):
    import tensorflow as tf
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential([tf.keras.Input(shape=(16,))] +
                               [tf.keras.layers.Dense(n, activation="relu") for n in hidden] +
                               [tf.keras.layers.Dense(7, activation="softmax")])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def fit_mlp(x, y, xv, yv, hidden, epochs, seed):
    """Use a bounded tf.data pool so CPU-only runs also work on small machines."""
    import tensorflow as tf
    model = build_mlp(hidden, seed)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    train = tf.data.Dataset.from_tensor_slices((x.astype("float32"), y)).shuffle(len(y), seed=seed).batch(32).with_options(options)
    val = tf.data.Dataset.from_tensor_slices((xv.astype("float32"), yv)).batch(256).with_options(options)
    callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)
    history = model.fit(train, validation_data=val, epochs=epochs, callbacks=[callback], verbose=0).history
    return model, {k: [float(v) for v in values] for k, values in history.items()}


def mlp_predictions(model, x):
    # Direct calls avoid creating an extra tf.data worker pool for inference.
    probabilities = np.concatenate([model(x[i:i+256].astype("float32"), training=False).numpy()
                                    for i in range(0, len(x), 256)])
    return np.asarray(CLASSES)[probabilities.argmax(axis=1)]


def save_figures(report, df, out):
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 10})
    fig, ax = plt.subplots(figsize=(9, 4))
    counts = df.Class.value_counts().reindex(CLASSES)
    ax.bar(CLASSES, counts, color="#168a91")
    ax.set(ylabel="Unique rows", title="Class distribution after exact deduplication")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout(); fig.savefig(out / "class_distribution.svg"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in ("MLP A", "MLP A unscaled"):
        h = report["models"][name]["history"]
        axes[0].plot(range(1, len(h["val_loss"])+1), h["val_loss"], label=name)
        axes[1].plot(range(1, len(h["val_accuracy"])+1), h["val_accuracy"], label=name)
    axes[0].set(yscale="log", ylabel="Validation cross-entropy (log)", xlabel="Epoch")
    axes[1].set(ylabel="Validation accuracy", xlabel="Epoch")
    for ax in axes: ax.legend(); ax.grid(alpha=.2)
    fig.suptitle("Effect of standardisation: same architecture and initial seed")
    fig.tight_layout(); fig.savefig(out / "scaling_experiment.svg"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    curve = report["training_size_experiment"]
    ax.plot([r["rows"] for r in curve], [r["validation"]["macro_f1"] for r in curve], marker="o")
    ax.set(xlabel="Training rows", ylabel="Validation macro F1", title="MLP A: effect of training data size (one seed)")
    ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(out / "training_size.svg"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 7))
    cm = np.asarray(report["selected_model_test"]["confusion_matrix"])
    image = ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(7), yticks=range(7), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Predicted", ylabel="True", title=report["selected_model"] + ": held-out test confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    for i in range(7):
        for j in range(7):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center", color="white" if cm[i,j] > cm.max()/2 else "black")
    fig.colorbar(image, ax=ax, shrink=.75)
    fig.tight_layout(); fig.savefig(out / "confusion_matrix.svg"); plt.close(fig)


def run(archive, out, epochs=30, seed=42):
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    df, source = load_data(archive)
    splits = split_data(df, seed)
    x = {k: df.loc[idx, FEATURES].to_numpy(dtype="float32") for k, idx in splits.items()}
    y = {k: df.loc[idx, "Class"].to_numpy() for k, idx in splits.items()}
    encoded = {k: np.searchsorted(CLASSES, labels).astype("int32") for k, labels in y.items()}
    scaler = StandardScaler().fit(x["train"])
    scaled = {k: scaler.transform(values) for k, values in x.items()}
    report = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
              "source": source, "features": FEATURES, "classes": CLASSES,
              "class_counts": {k: int(v) for k,v in df.Class.value_counts().items()},
              "splits": {k: {"rows": len(v), "source_row_indices_sha256": hashlib.sha256(np.asarray(v, dtype="<i8").tobytes()).hexdigest()} for k,v in splits.items()},
              "versions": {p: version(p) for p in ("tensorflow", "keras", "numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "joblib")},
              "python": platform.python_version(), "platform": platform.platform(),
              "config": {"max_epochs": epochs, "batch_size": 32, "learning_rate": .001, "early_stopping_patience": 7},
              "models": {}, "training_size_experiment": []}
    classical = {"Majority baseline": DummyClassifier(strategy="most_frequent"),
                 "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)),
                 "RBF SVM": make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf")),
                 "Random forest": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=2)}
    predictions = {}; fitted = {}
    for name, model in classical.items():
        model.fit(x["train"], y["train"])
        report["models"][name] = {"validation": scores(y["validation"], model.predict(x["validation"]))}
        fitted[name] = model
        print(name, report["models"][name]["validation"], flush=True)
    for name, hidden in [*ARCHITECTURES.items(), ("MLP A unscaled", [64, 32])]:
        values = x if name.endswith("unscaled") else scaled
        model, history = fit_mlp(values["train"], encoded["train"], values["validation"], encoded["validation"], hidden, epochs, seed)
        report["models"][name] = {"hidden_layers": hidden, "parameters": model.count_params(), "history": history,
                                  "best_epoch": int(np.argmin(history["val_loss"]))+1,
                                  "validation": scores(y["validation"], mlp_predictions(model, values["validation"]))}
        model.save(out / (name.replace(" ", "_") + ".keras"))
        print(name, report["models"][name]["validation"], flush=True)
    # All candidates and selection rules are fixed before test predictions are made.
    eligible = [n for n in report["models"] if n != "MLP A unscaled"]
    selected = max(eligible, key=lambda n: report["models"][n]["validation"]["macro_f1"])
    report["selected_model"] = selected
    report["selection_rule"] = "Highest validation macro F1; unscaled ablation excluded; no refit on validation"
    for name in report["models"]:
        if name in fitted:
            pred = fitted[name].predict(x["test"])
        else:
            model = tf.keras.models.load_model(out / (name.replace(" ", "_") + ".keras"))
            pred = mlp_predictions(model, x["test"] if name.endswith("unscaled") else scaled["test"])
        predictions[name] = pred
        report["models"][name]["test"] = scores(y["test"], pred)
    pred = predictions[selected]
    report["selected_model_test"] = {"classification_report": classification_report(y["test"], pred, labels=CLASSES, output_dict=True, zero_division=0),
                                     "confusion_matrix": confusion_matrix(y["test"], pred, labels=CLASSES).tolist()}
    errors = pd.DataFrame({"source_row": splits["test"], "true": y["test"], "predicted": pred})
    errors[errors.true != errors.predicted].to_csv(out / "test_errors.csv", index=False)
    examples = df.loc[splits["test"]].groupby("Class", sort=True).head(1)
    examples[FEATURES].to_csv(out / "example_features.csv", index=False)
    prediction_by_row = dict(zip(splits["test"], pred))
    report["example_predictions"] = [{"csv_row": i, "true": row.Class, "predicted": prediction_by_row[index]}
                                     for i, (index, row) in enumerate(examples.iterrows())]
    joblib.dump({"name": selected, "features": FEATURES, "scaler": scaler,
                 "model": fitted.get(selected), "keras_file": None if selected in fitted else selected.replace(" ", "_")+".keras"}, out / "predictor.joblib")
    # Learning curve uses validation only; each subset fits its own scaler.
    for fraction in [.2, .4, .6, .8, 1.0]:
        if fraction == 1.0:
            row = {"fraction": fraction, "rows": len(x["train"]), "validation": report["models"]["MLP A"]["validation"]}
        else:
            idx, _ = train_test_split(np.arange(len(x["train"])), train_size=fraction, random_state=seed, stratify=encoded["train"])
            sub_scaler = StandardScaler().fit(x["train"][idx])
            model, history = fit_mlp(sub_scaler.transform(x["train"][idx]), encoded["train"][idx], sub_scaler.transform(x["validation"]), encoded["validation"], [64,32], epochs, seed)
            row = {"fraction": fraction, "rows": len(idx), "best_epoch": int(np.argmin(history["val_loss"]))+1,
                   "validation": scores(y["validation"], mlp_predictions(model, sub_scaler.transform(x["validation"])))}
        report["training_size_experiment"].append(row)
        print("Training fraction", row, flush=True)
    (out / "metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    save_figures(report, df, out)
    print("Selected:", selected, report["models"][selected]["test"], flush=True)
    return report


def predict(predictor, rows):
    predictor = Path(predictor)
    bundle = joblib.load(predictor)
    if set(rows.columns) != set(bundle["features"]):
        raise ValueError("Prediction CSV must contain exactly the 16 documented feature columns")
    x = rows[bundle["features"]].to_numpy(dtype="float32")
    if len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("Prediction rows must contain finite numeric values")
    if bundle["keras_file"]:
        import tensorflow as tf
        model = tf.keras.models.load_model(predictor.parent / bundle["keras_file"])
        return mlp_predictions(model, bundle["scaler"].transform(x))
    return bundle["model"].predict(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--data", default="data/dry-bean.zip")
    train.add_argument("--output", default="runs/full")
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--seed", type=int, default=42)
    inference = commands.add_parser("predict")
    inference.add_argument("--predictor", required=True)
    inference.add_argument("--csv", required=True)
    args = parser.parse_args()
    if args.command == "train":
        if args.epochs < 1: parser.error("--epochs must be positive")
        run(args.data, args.output, args.epochs, args.seed)
    else:
        print(json.dumps(predict(args.predictor, pd.read_csv(args.csv)).tolist()))
