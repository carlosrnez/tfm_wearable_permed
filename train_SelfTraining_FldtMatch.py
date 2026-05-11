import sys
import time
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_DATA     = "arr_0"
WINDOW_LABELS   = "arr_1"
WINDOW_METADATA = "arr_2"

ACTIVITIES_TO_REMOVE = {"TAPIZ RODANTE", "YOGA"}

SUPERCLASSES_CAPTURED24 = sorted([
    "WALKING", "HOUSEHOLD-CHORES", "STANDING", "SLEEP",
    "BICYCLING", "SITTING", "MIXED-ACTIVITY", "SPORTS",
])

MAPPING_CAPTURED24: dict[str, str] = {}   # pendiente de rellenar

SUPERCLASSES_CPA_METS = ["SEDENTARY", "LIGHT-INTENSITY", "MODERATE-INTENSITY", "VIGOROUS-INTENSITY"]

MAPPING_CPA_METS: dict[str, str] = {
    # SEDENTARY
    "FASE REPOSO CON K5":   "SEDENTARY",
    "SENTADO LEYENDO":       "SEDENTARY",
    "SENTADO USANDO PC":     "SEDENTARY",
    "SENTADO VIENDO LA TV":  "SEDENTARY",
    # LIGHT-INTENSITY
    "DE PIE DOBLANDO TOALLAS":   "LIGHT-INTENSITY",
    "DE PIE USANDO PC":          "LIGHT-INTENSITY",
    "CAMINAR CON MÓVIL O LIBRO": "LIGHT-INTENSITY",
    "CAMINAR ZIGZAG":            "LIGHT-INTENSITY",
    # MODERATE-INTENSITY
    "DE PIE BARRIENDO":       "MODERATE-INTENSITY",
    "DE PIE MOVIENDO LIBROS": "MODERATE-INTENSITY",
    "CAMINAR CON LA COMPRA":  "MODERATE-INTENSITY",
    "CAMINAR USUAL SPEED":    "MODERATE-INTENSITY",
    "SUBIR Y BAJAR ESCALERAS":"MODERATE-INTENSITY",
    # VIGOROUS-INTENSITY
    "INCREMENTAL CICLOERGOMETRO": "VIGOROUS-INTENSITY",
    "TROTAR":                     "VIGOROUS-INTENSITY",
}

SUPERCLASS_REGISTRY = {
    "Captured24": (SUPERCLASSES_CAPTURED24, MAPPING_CAPTURED24),
    "CPA-METS":   (SUPERCLASSES_CPA_METS,  MAPPING_CPA_METS),
}

SAVE_PATH_BY_SUPERCLASS = {
    "CPA-METS":   "4_classes",
    "Captured24": "8_classes",
}

UNLABELED_LABELS = {"ACTIVIDAD NO ESTRUCTURADA", "UNKNOWN"}

# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args(args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autoencoder + Self-Training with dynamic SDT thresholds"
    )
    parser.add_argument("--stack-all",        required=True, dest="stack_all")
    parser.add_argument("--sensor",           required=True, choices=["PI", "M"])
    parser.add_argument("--superclases",      dest="superclases", default=None)
    parser.add_argument("--loops",            dest="loops",           type=int,   default=30)
    parser.add_argument("--optimize-trials",  dest="optimize_trials", type=int,   default=1)
    parser.add_argument("--lambda",           dest="lambda_",         type=float, default=10.0,
                        help="SDT shield parameter λ (>1). Default: 10.")
    parser.add_argument("--gamma",            dest="gamma",           type=float, default=0.95,
                        help="SDT base threshold γ ∈ (0,1]. Default: 0.95.")
    parser.add_argument("--generate-plots",   dest="generate_plots",  action="store_true", default=False)
    parser.add_argument("-v",  "--verbose",      dest="loglevel", action="store_const", const=logging.INFO)
    parser.add_argument("-vv", "--very-verbose", dest="loglevel", action="store_const", const=logging.DEBUG)
    return parser.parse_args(args)

# ── Data helpers ──────────────────────────────────────────────────────────────
def load_dataset(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stack = np.load(path)
    return stack[WINDOW_DATA], stack[WINDOW_LABELS], stack[WINDOW_METADATA]


def remove_activities(
    X: np.ndarray, y: np.ndarray, m: np.ndarray, to_remove: set[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = ~np.isin(y, list(to_remove))
    return X[mask], y[mask], m[mask]


def apply_superclass_mapping(y: np.ndarray, mapping: dict[str, str]) -> np.ndarray:
    return np.array([mapping.get(label, "UNKNOWN") for label in y])


def participant_loocv_iterator(
    X: np.ndarray, y: np.ndarray, m: np.ndarray, val_size: float = 0.2
):
    """Leave-One-Group-Out for test; GroupShuffleSplit for validation."""
    logo    = LeaveOneGroupOut()
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size)

    for train_val_idx, test_idx in logo.split(X, y, groups=m):
        X_tv, X_test = X[train_val_idx], X[test_idx]
        y_tv, y_test = y[train_val_idx], y[test_idx]
        m_tv, m_test = m[train_val_idx], m[test_idx]

        try:
            train_idx, val_idx = next(gss_val.split(X_tv, y_tv, groups=m_tv))
        except StopIteration:
            continue

        log.info(
            "Fold — test participant(s): %s | train groups: %d | val groups: %d",
            np.unique(m_test),
            len(np.unique(m_tv[train_idx])),
            len(np.unique(m_tv[val_idx])),
        )
        yield (
            X_tv[train_idx], X_tv[val_idx], X_test,
            y_tv[train_idx], y_tv[val_idx], y_test,
            m_tv[train_idx], m_tv[val_idx], m_test,
        )

# ── Autoencoder ───────────────────────────────────────────────────────────────
def build_autoencoder(
    input_dim: int, latent_dim: int, dropout: float = 0.0, l2_reg: float = 0.0
) -> tuple[models.Model, models.Model]:
    """Returns (autoencoder, encoder)."""
    inp     = layers.Input(shape=(input_dim,), name="input")
    x       = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="enc_128")(inp)
    x       = layers.Dropout(dropout)(x)
    x       = layers.Dense(64,  activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="enc_64")(x)
    latent  = layers.Dense(latent_dim, activation="linear", kernel_regularizer=regularizers.l2(l2_reg), name="latent")(x)
    x       = layers.Dense(64,  activation="relu", name="dec_64")(latent)
    x       = layers.Dense(128, activation="relu", name="dec_128")(x)
    out     = layers.Dense(input_dim, activation="linear", name="output")(x)

    return models.Model(inp, out), models.Model(inp, latent)


def _optuna_objective(trial, X_train: np.ndarray, X_val: np.ndarray) -> float:
    params = {
        "latent_dim": trial.suggest_int("latent_dim", 4, 64),
        "dropout":    trial.suggest_float("dropout", 0.0, 0.5),
        "l2_reg":     trial.suggest_float("l2_reg", 1e-6, 1e-2, log=True),
        "lr":         trial.suggest_float("lr",     1e-5, 1e-2, log=True),
    }
    ae, _ = build_autoencoder(X_train.shape[1], **params)
    ae.compile(optimizer=Adam(params["lr"]), loss="mse")
    history = ae.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=50, batch_size=64,
        callbacks=[EarlyStopping(patience=5)],
        verbose=0,
    )
    return min(history.history["val_loss"])


def optimize_and_train_autoencoder(
    X_train: np.ndarray, X_val: np.ndarray, n_trials: int
) -> tuple[models.Model, models.Model]:
    """
    Runs Optuna hyperparameter search, then re-trains the best autoencoder
    from scratch on the full training set.
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: _optuna_objective(trial, X_train, X_val), n_trials=n_trials)

    best = study.best_trial.params
    log.info("Best hyperparameters: %s", best)

    clear_session()
    ae, encoder = build_autoencoder(
        input_dim=X_train.shape[1],
        latent_dim=best["latent_dim"],
        dropout=best["dropout"],
        l2_reg=best["l2_reg"],
    )
    ae.compile(optimizer=Adam(learning_rate=best["lr"]), loss="mse")
    ae.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=200, batch_size=64,
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=0,
    )
    return ae, encoder

# ── SDT ───────────────────────────────────────────────────────────────────────
def compute_sdt_thresholds(
    y_labeled: np.ndarray, lambda_: float = 10.0, gamma: float = 0.95
) -> dict[str, float]:
    """
    Dynamic per-class threshold (FldtMatch SDT formula).

    Parameters
    ----------
    y_labeled : current labeled set labels (grows with pseudo-labels).
    lambda_   : shield parameter (>1). Higher → minority classes protected.
    gamma     : base threshold; the majority class always receives this value.

    Returns
    -------
    {class_name: threshold}
    """
    classes, counts = np.unique(y_labeled, return_counts=True)
    n_max = counts.max()

    return {
        cls: float(np.clip(np.log(n_max / n_c + lambda_ - 1) / np.log(lambda_) * gamma, 0.0, 1.0))
        for cls, n_c in zip(classes, counts)
    }

# ── Self-Training ─────────────────────────────────────────────────────────────
def run_self_training_sdt(
    clf: RandomForestClassifier,
    Z_labeled: np.ndarray,
    y_labeled: np.ndarray,
    Z_unlabeled: np.ndarray,
    lambda_: float,
    gamma: float,
) -> RandomForestClassifier:
    """
    Iterative self-training with per-class dynamic SDT thresholds.
    Returns the classifier retrained on labeled + accepted pseudo-labeled data.
    """
    log.info("Self-Training SDT start — unlabeled pool: %d samples", len(Z_unlabeled))

    for iteration in range(1, sys.maxsize):
        if len(Z_unlabeled) == 0:
            break

        thresholds = compute_sdt_thresholds(y_labeled, lambda_=lambda_, gamma=gamma)

        if iteration == 1:
            log.info("Initial SDT thresholds: %s", {k: f"{v:.4f}" for k, v in sorted(thresholds.items())})

        probas       = clf.predict_proba(Z_unlabeled)
        pred_indices = np.argmax(probas, axis=1)
        pred_classes = clf.classes_[pred_indices]
        max_probas   = probas[np.arange(len(probas)), pred_indices]

        # Vectorised per-class threshold comparison
        threshold_per_sample = np.array([thresholds.get(c, gamma) for c in pred_classes])
        confident_mask       = max_probas >= threshold_per_sample

        n_accepted = confident_mask.sum()
        if n_accepted == 0:
            log.info("Iteration %d: no sample exceeded its SDT threshold — stopping.", iteration)
            break

        log.info("Iteration %d: +%d samples accepted (%d remaining in pool).",
                 iteration, n_accepted, len(Z_unlabeled) - n_accepted)
        for cls, cnt in zip(*np.unique(pred_classes[confident_mask], return_counts=True)):
            log.debug("  · %s: %d samples (threshold=%.4f)", cls, cnt, thresholds.get(cls, gamma))

        Z_labeled   = np.vstack([Z_labeled,   Z_unlabeled[confident_mask]])
        y_labeled   = np.concatenate([y_labeled, pred_classes[confident_mask]])
        Z_unlabeled = Z_unlabeled[~confident_mask]

        clf.fit(Z_labeled, y_labeled)

    return clf

# ── Single fold ───────────────────────────────────────────────────────────────
def run_fold(
    fold_idx: int,
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
    n_trials: int, lambda_: float, gamma: float,
) -> dict:
    log.info("══ Fold %d ══", fold_idx)
    t0 = time.perf_counter()

    # 1. Fit scaler on train only (no leakage)
    scaler  = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # 2. Optimize & train autoencoder
    log.info("Optimising autoencoder hyperparameters (%d trial(s))", n_trials)
    _, encoder = optimize_and_train_autoencoder(X_train, X_val, n_trials)

    # 3. Encode
    Z_train = encoder.predict(X_train, verbose=0)
    Z_test  = encoder.predict(X_test,  verbose=0)

    # 4. Split labeled / unlabeled
    labeled_mask = ~np.isin(y_train, list(UNLABELED_LABELS))
    Z_labeled, y_labeled     = Z_train[labeled_mask],  y_train[labeled_mask]
    Z_unlabeled              = Z_train[~labeled_mask]

    # 5. Baseline classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(Z_labeled, y_labeled)

    # 6. Self-Training with SDT
    clf = run_self_training_sdt(clf, Z_labeled, y_labeled, Z_unlabeled, lambda_, gamma)

    # 7. Evaluate (exclude unlabeled samples from test)
    test_labeled_mask = ~np.isin(y_test, list(UNLABELED_LABELS))
    y_true = y_test[test_labeled_mask]
    y_pred = clf.predict(Z_test[test_labeled_mask])

    elapsed = time.perf_counter() - t0
    log.info("Fold %d done in %.1fs — accuracy=%.4f  macro-F1=%.4f",
             fold_idx, elapsed,
             accuracy_score(y_true, y_pred),
             f1_score(y_true, y_pred, average="macro"))

    return {
        "loop":                      fold_idx,
        "classifier_model_accuracy": accuracy_score(y_true, y_pred),
        "classifier_model_f1_score": f1_score(y_true, y_pred, average="macro"),
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])

    if args.loglevel:
        logging.getLogger().setLevel(args.loglevel)

    t_start = time.perf_counter()

    # ── Load ──────────────────────────────────────────────────────────────────
    log.info("Loading dataset: %s", args.stack_all)
    X, y, m = load_dataset(args.stack_all)

    # ── Pre-process ───────────────────────────────────────────────────────────
    log.info("Removing activities: %s", ACTIVITIES_TO_REMOVE)
    X, y, m = remove_activities(X, y, m, ACTIVITIES_TO_REMOVE)

    if args.superclases in SUPERCLASS_REGISTRY:
        _, mapping = SUPERCLASS_REGISTRY[args.superclases]
        log.info("Applying superclass mapping: %s", args.superclases)
        y = apply_superclass_mapping(y, mapping)

    # ── LOOCV ─────────────────────────────────────────────────────────────────
    log.info("Starting LOOCV (λ=%.2f, γ=%.2f)", args.lambda_, args.gamma)
    metrics = []

    for fold_idx, (X_tr, X_val, X_te, y_tr, y_val, y_te, *_) in enumerate(
        participant_loocv_iterator(X, y, m), start=1
    ):
        metric = run_fold(
            fold_idx,
            X_tr, X_val, X_te,
            y_tr, y_val, y_te,
            n_trials=args.optimize_trials,
            lambda_=args.lambda_,
            gamma=args.gamma,
        )
        metrics.append(metric)

    # ── Metrics summary ───────────────────────────────────────────────────────
    df = pd.DataFrame(metrics)
    summary = df.agg(["mean", "std"]).reset_index().rename(columns={"index": "loop"})
    df = pd.concat([df, summary], ignore_index=True)

    save_dir = Path.cwd() / "reports"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"metrics_SelfTraining_SAT_{args.sensor}.csv"
    df.to_csv(save_path, index=False)
    log.info("Metrics saved to: %s", save_path)

    log.info("Total time: %.1fs", time.perf_counter() - t_start)


if __name__ == "__main__":
    main()