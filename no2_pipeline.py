"""Data loading, features, splits, models and metrics for the NO2 level project.

Task: predict the NO2 concentration level (5 equal-frequency classes) from the
other sensors and weather variables of the UCI Air Quality dataset.

Everything fitted on data (class edges, medians, scaler) uses the training part
only. Lag features use past values only.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET = "NO2(GT)"
N_BINS = 5


# --------------------------------------------------------------------------
# Data and features
# --------------------------------------------------------------------------
def load_data(csv_path):
    """Read the CSV and return (dataframe with labelled rows, feature sets)."""
    df = pd.read_csv(csv_path, sep=";")
    df = df.drop(columns=["Unnamed: 15", "Unnamed: 16", "NMHC(GT)"], errors="ignore")
    df = df.dropna(how="all")
    for col in df.columns:  # decimal commas -> floats
        if col not in ("Date", "Time") and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ".", regex=False),
                                    errors="coerce")
    df = df.replace(-200, np.nan)  # -200 marks a missing or corrupted reading
    df["dt"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H.%M.%S")
    df = df.sort_values("dt").reset_index(drop=True)

    gt_cols = [c for c in df.columns if c.endswith("(GT)") and c != TARGET]
    sensor_cols = [c for c in df.columns
                   if c not in gt_cols + ["Date", "Time", "dt", TARGET]]

    hour, dow = df["dt"].dt.hour, df["dt"].dt.dayofweek
    df["hour_raw"], df["dow_raw"] = hour, dow
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    df["dow_sin"], df["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    cyclical = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]

    # Lags are built on the full hourly series BEFORE dropping rows without a
    # target, and only use past values (shift(1)).
    lags = []
    for c in sensor_cols:
        df[c + "_lag1"] = df[c].shift(1)
        df[c + "_mean3"] = df[c].shift(1).rolling(3, min_periods=1).mean()
        lags += [c + "_lag1", c + "_mean3"]

    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    feature_sets = {
        "all": gt_cols + sensor_cols,
        "all_rawtime": gt_cols + sensor_cols + ["hour_raw", "dow_raw"],
        "all_time_lags": gt_cols + sensor_cols + cyclical + lags,
        "sensors": sensor_cols,  # no reference-analyzer "(GT)" columns
        "sensors_time_lags": sensor_cols + cyclical + lags,
    }
    return df, feature_sets


def make_split(df, mode, seed):
    """Return (train_idx, test_idx).

    random : random hours (neighbouring hours can end up on both sides)
    block  : random whole weeks (no hourly leakage, no systematic shift)
    chrono : the last 20% of the timeline is the test set
    """
    n = len(df)
    if mode == "random":
        return train_test_split(np.arange(n), test_size=0.2, random_state=seed)
    if mode == "block":
        week = ((df["dt"] - df["dt"].min()).dt.days // 7).to_numpy()
        weeks = np.unique(week)
        test_weeks = np.random.default_rng(seed).choice(
            weeks, size=int(0.2 * len(weeks)), replace=False)
        is_test = np.isin(week, test_weeks)
        return np.where(~is_test)[0], np.where(is_test)[0]
    if mode == "chrono":
        cut = int(n * 0.8)
        return np.arange(cut), np.arange(cut, n)
    raise ValueError(f"unknown split: {mode}")


def class_edges(y_train):
    """Quantile edges computed on the training part only."""
    return np.quantile(y_train, np.linspace(0, 1, N_BINS + 1)[1:-1])


def preprocess(X_train, X_test):
    """Median imputation and standard scaling, fitted on the training part only."""
    medians = X_train.median()
    X_train, X_test = X_train.fillna(medians), X_test.fillna(medians)
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def build_mlp(n_inputs, regularized):
    """Feedforward network with three hidden layers (128-64-32)."""
    from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    if regularized:
        model = Sequential([
            Dense(128, activation="relu", input_shape=(n_inputs,)),
            BatchNormalization(), Dropout(0.2),
            Dense(64, activation="relu"),
            BatchNormalization(), Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(N_BINS, activation="softmax"),
        ])
        optimizer = Adam(learning_rate=0.0005)
    else:
        model = Sequential([
            Dense(128, activation="relu", input_shape=(n_inputs,)),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(N_BINS, activation="softmax"),
        ])
        optimizer = Adam()
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def fit_model(name, X_train, y_cont_train, y_cls_train, edges, seed):
    """Fit a model; return a function mapping features to predicted class ids.

    mlp_plain / mlp_reg : feedforward network without / with BatchNorm + Dropout
    hgb_clf             : gradient boosting classifier
    hgb_reg             : gradient boosting regressor, predictions binned with
                          the training edges
    """
    if name in ("mlp_plain", "mlp_reg"):
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping
        tf.random.set_seed(seed)
        np.random.seed(seed)
        model = build_mlp(X_train.shape[1], regularized=(name == "mlp_reg"))
        stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
        model.fit(X_train, y_cls_train, epochs=100, batch_size=32,
                  validation_split=0.2, callbacks=[stop], verbose=0)
        return lambda X: model.predict(X, verbose=0).argmax(axis=1)
    if name == "hgb_clf":
        model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, random_state=seed).fit(X_train, y_cls_train)
        return model.predict
    if name == "hgb_reg":
        model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, random_state=seed).fit(X_train, y_cont_train)
        return lambda X: np.digitize(model.predict(X), edges)
    raise ValueError(f"unknown model: {name}")


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate(y_true, y_pred):
    """Accuracy, macro-F1 and ordinal error statistics for class ids."""
    d = np.abs(y_true - y_pred)
    n_err = int((d > 0).sum())
    return {
        "acc": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "within1": float((d <= 1).mean()),                # off by at most one class
        "mae_class": float(d.mean()),                     # mean class distance
        "adj_err_share": float((d == 1).sum() / n_err) if n_err else np.nan,
        "under_err_share": float((y_pred < y_true).sum() / n_err) if n_err else np.nan,
    }


def permutation_importance(predict, X, y, names, n_repeats=5, seed=0):
    """Drop in macro-F1 when one feature column is shuffled (on held-out data)."""
    rng = np.random.default_rng(seed)
    base = f1_score(y, predict(X), average="macro", zero_division=0)
    drops = {}
    for j, name in enumerate(names):
        scores = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            scores.append(base - f1_score(y, predict(X_perm), average="macro", zero_division=0))
        drops[name] = float(np.mean(scores))
    return pd.Series(drops, name="macro_f1_drop").sort_values(ascending=False)
