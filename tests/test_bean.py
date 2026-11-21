import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler

from bean import CLASSES, FEATURES, build_mlp, clean_data, fit_mlp, mlp_predictions, predict, split_data


def sample_data():
    rng = np.random.default_rng(7)
    df = pd.DataFrame(rng.normal(size=(210, 16)), columns=FEATURES)
    df["Class"] = np.repeat(CLASSES, 30)
    return df


def test_source_validation_and_deduplication():
    source = sample_data()
    duplicated = pd.concat([source, source.iloc[:2]], ignore_index=True)
    assert len(clean_data(duplicated)) == len(source)
    missing = source.copy(); missing.loc[0, "Area"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        clean_data(missing)
    conflict = source.iloc[:1].copy(); conflict["Class"] = "SIRA"
    with pytest.raises(ValueError, match="conflicting"):
        clean_data(pd.concat([source, conflict]))


def test_splits_are_disjoint_complete_and_reproducible():
    data = sample_data()
    first, second = split_data(data), split_data(data)
    groups = [set(v) for v in first.values()]
    assert set.union(*groups) == set(data.index)
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
        assert set(data.loc[first[key], "Class"]) == set(CLASSES)


def test_prediction_respects_names_and_rejects_invalid_input(tmp_path):
    rows = sample_data()[FEATURES]
    model = DummyClassifier(strategy="constant", constant="SEKER").fit(rows.to_numpy(), np.repeat("SEKER", len(rows)))
    artifact = tmp_path / "predictor.joblib"
    joblib.dump({"features": FEATURES, "model": model, "keras_file": None}, artifact)
    assert predict(artifact, rows[FEATURES[::-1]])[0] == "SEKER"
    with pytest.raises(ValueError, match="16"):
        predict(artifact, rows.drop(columns="Area"))
    invalid = rows.copy(); invalid.loc[0, "Area"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        predict(artifact, invalid)


def test_keras_training_and_checkpoint_roundtrip(tmp_path):
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    df = sample_data()
    x = StandardScaler().fit_transform(df[FEATURES]).astype("float32")
    y = np.searchsorted(CLASSES, df.Class).astype("int32")
    model, history = fit_mlp(x, y, x[:21], y[:21], [16], 1, 42)
    assert np.isfinite(history["loss"]).all()
    assert model.count_params() == 391
    before = mlp_predictions(model, x[:8])
    checkpoint = tmp_path / "model.keras"
    model.save(checkpoint)
    restored = tf.keras.models.load_model(checkpoint)
    np.testing.assert_array_equal(mlp_predictions(restored, x[:8]), before)
