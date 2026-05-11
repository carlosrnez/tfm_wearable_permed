import sys
import time
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import optuna
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import EarlyStopping

ACTIVITIES = sorted(['FASE REPOSO CON K5', 'TAPIZ RODANTE',
                     'CAMINAR ZIGZAG', 'TROTAR', 'SUBIR Y BAJAR ESCALERAS'])

ACTIVITIES_TO_BE_REMOVED=['TAPIZ RODANTE', 'YOGA']

SUPERCLASES_CAPTURED24 = sorted(['WALKING', 'HOUSEHOLD-CHORES',
                                 'STANDING', 'SLEEP', 'BICYCLING',
                                 'SITTING', 'MIXED-ACTIVITY', 'SPORTS'])

MAPPING_CAPTURED24 = {
}

SUPERCLASES_CPA_METS = ['SEDENTARY', 'LIGHT-INTENSITY',
                        'MODERATE-INTENSITY', 'VIGOROUS-INTENSITY']

MAPPING_CPA_METS = {
    # SEDENTARY
    'FASE REPOSO CON K5': 'SEDENTARY',
    'SENTADO LEYENDO': 'SEDENTARY',
    'SENTADO USANDO PC': 'SEDENTARY',
    'SENTADO VIENDO LA TV': 'SEDENTARY',    

    # LIGHT-INTENSITY
    'DE PIE DOBLANDO TOALLAS': 'LIGHT-INTENSITY',
    'DE PIE USANDO PC': 'LIGHT-INTENSITY',
    'CAMINAR CON MÓVIL O LIBRO': 'LIGHT-INTENSITY',
    'CAMINAR ZIGZAG': 'LIGHT-INTENSITY',
    
    # MODERATE-INTENSITY
    'DE PIE BARRIENDO': 'MODERATE-INTENSITY',
    'DE PIE MOVIENDO LIBROS': 'MODERATE-INTENSITY',
    'CAMINAR CON LA COMPRA': 'MODERATE-INTENSITY',
    'CAMINAR USUAL SPEED': 'MODERATE-INTENSITY',
    'SUBIR Y BAJAR ESCALERAS': 'MODERATE-INTENSITY',

    # VIGOROUS-INTENSITY
    'INCREMENTAL CICLOERGOMETRO': 'VIGOROUS-INTENSITY',  
    'TROTAR': 'VIGOROUS-INTENSITY' 
}

WINDOW_DATA = "arr_0"
WINDOW_LABELS = "arr_1"
WINDOW_METADATA = "arr_2"

metrics = []

def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Mixture of Experts with autoencoders experts")

    parser.add_argument(
        "-stack-all",
        "--stack-all",
        required=True,        
        dest="stack_all",       
        help=f"Participant stack data."        
    )

    parser.add_argument(
        "-sensor",
        "--sensor",
        required=True,
        choices=["PI", "M"],
        dest="sensor",
        help="Indica si el dataset es de Pierna (PI) o Muñeca (M)."
    )

    parser.add_argument(
        "-superclases",
        "--superclases",
        dest="superclases",        
        help=f"Use Superclases: WearablePerMed, Captured24, CPA-METS"
    )
    parser.add_argument(
        "-loops",
        "--loops",
        dest="loops",        
        type=int,        
        default=30,        
        help="Number of loops."
    )     
    parser.add_argument(
        "-optimize-trials",
        "--optimize-trials",               
        dest="optimize_trials",
        type=int,
        default=1,               
        help=f"Optimize hyperparameters num trials."        
    )
    parser.add_argument(
        '-generate-plots',
        '--generate-plots',
        dest='generate_plots',
        action='store_true',
        default=False,
        help="Generate Plots"
    )                   
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        action="store_const",
        const=logging.INFO,
        help="set log level to verbose."
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        action="store_const",
        const=logging.DEBUG,
        help="set log level to very verbose."        
    )

    return parser.parse_args(args)

def get_save_path(superclases):
    if superclases == 'CPA-METS':
        return '4_classes'
    elif superclases == 'Captured24':
        return '8_classes'
    else:
        return '15_classes'
    
def pretreatment(y_data):
    # Get indices of elements to remove
    indices_to_remove = [i for i, lbl in enumerate(y_data) if lbl in ACTIVITIES_TO_BE_REMOVED]

    return indices_to_remove

def superclases_captured24(y_data):
    return np.array([MAPPING_CAPTURED24.get(label, "UNKNOWN") for label in y_data])

def superclases_cpa_mets(y_data):
    return np.array([MAPPING_CPA_METS.get(label, "UNKNOWN") for label in y_data])

def participant_loocv_iterator(X_data, y_data, m_data, val_size=0.2):
    """
    Generator that performs Leave-One-Group-Out for testing, 
    and uses GroupShuffleSplit to create a validation set from the remainder.
    """
    logo = LeaveOneGroupOut()

    # Outer Loop: Leave one participant out for TEST
    for train_val_idx, test_idx in logo.split(X_data, y_data, groups=m_data):
        
        X_train_val, X_test = X_data[train_val_idx], X_data[test_idx]
        y_train_val, y_test = y_data[train_val_idx], y_data[test_idx]
        m_train_val, m_test = m_data[train_val_idx], m_data[test_idx]

        # Inner Split: Get VALIDATION from the remaining participants
        # We use GroupShuffleSplit to ensure the val participant isn't in train
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size)
        
        try:
            train_idx, val_idx = next(gss_val.split(X_train_val, y_train_val, groups=m_train_val))
        except StopIteration:
            # Handle cases where there aren't enough groups left to split
            continue

        X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
        y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
        m_train, m_val = m_train_val[train_idx], m_train_val[val_idx]

        # Log the current fold status
        print(f"--- Fold for Participant(s) {np.unique(m_test)} ---")
        print(f"Train Groups: {len(np.unique(m_train))} | Val Groups: {len(np.unique(m_val))}")

        # Yield the data so the loop can process one fold at a time
        yield (
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            m_train, m_val, m_test
        )

def build_autoencoder(input_dim, latent_dim, dropout=0.0, l2_reg=0.0):
    # Encoder
    input_layer = layers.Input(shape=(input_dim,), name="input")    
    encoded = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="enc_dense_128")(input_layer)
    encoded = layers.Dropout(dropout)(encoded)
    encoded = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="enc_dense_64")(encoded)

    latent = layers.Dense(latent_dim, activation="linear", kernel_regularizer=regularizers.l2(l2_reg), name="latent")(encoded)

    # Decoder
    decoded = layers.Dense(64, activation="relu", name="dec_dense_64")(latent)
    decoded = layers.Dense(128, activation="relu", name="dec_dense_128")(decoded)
    output_layer = layers.Dense(input_dim, activation="linear", name="output")(decoded)

    autoencoder = models.Model(input_layer, output_layer)
    encoder = models.Model(input_layer, latent)

    return autoencoder, encoder

def objective(trial, X_train, X_validation):
    latent_dim = trial.suggest_int("latent_dim", 4, 64)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    l2_reg = trial.suggest_float("l2_reg", 1e-6, 1e-2, log=True) # L2 should ALWAYS be log-scaled
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    
    autoencoder, encoder = build_autoencoder(X_train.shape[1], latent_dim, dropout, l2_reg)

    autoencoder.compile(
        optimizer=Adam(lr),
        loss="mse"
    )

    history = autoencoder.fit(
        X_train, X_train,
        validation_data=(X_validation, X_validation),
        epochs=50,
        batch_size=64,
        callbacks=[EarlyStopping(patience=5)],
        verbose=0
    )

    return min(history.history["val_loss"])

start_app = time.perf_counter()

args = parse_args(sys.argv[1:])

print("🟢 Loading dataset")
stack_data_all = np.load(args.stack_all)

X_data_all = stack_data_all[WINDOW_DATA]
y_data_all = stack_data_all[WINDOW_LABELS]
m_data_all = stack_data_all[WINDOW_METADATA]

print("🟢 Remove some activities")
ACTIVITIES = [x for x in ACTIVITIES if x not in ACTIVITIES_TO_BE_REMOVED]

indices_to_remove = pretreatment(y_data_all)

X_data = np.delete(X_data_all, indices_to_remove, axis=0)
y_data = np.delete(y_data_all, indices_to_remove, axis=0)
m_data = np.delete(m_data_all, indices_to_remove, axis=0)

print("🟢 Regroup labels vector")

if (args.superclases == "Captured24"):
    ACTIVITIES = SUPERCLASES_CAPTURED24
    (y_data) = superclases_captured24(y_data)    
elif (args.superclases == "CPA-METS"):
    ACTIVITIES = SUPERCLASES_CPA_METS
    (y_data) = superclases_cpa_mets(y_data)

print("🟢 Standarize data") 
sc = StandardScaler()

X_data = sc.fit_transform(X_data)

print("Calculate LOOCV(Leave-One-Out)")
data_iterator = participant_loocv_iterator(X_data, y_data, m_data)

loops = []

for loop, (X_train, X_validation, X_test, y_train, y_validation, y_test, m_train, m_validation, m_test) in enumerate(data_iterator, start=1):    
    start_loop = time.perf_counter()
        
    metric = {}

    print("🔵 Loop: " + str(loop))
    loops.append(loop)

    print("🟢 Optimize Autoencoder hyperparameters")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train, X_validation), n_trials=args.optimize_trials)

    best_params = study.best_trial.params
    print(best_params)

    print("🟢 Build Autoencoder with best parameters")
    clear_session()

    autoencoder, encoder = build_autoencoder(input_dim=X_train.shape[1], latent_dim=best_params["latent_dim"], dropout=best_params["dropout"])

    autoencoder.compile(optimizer=Adam(learning_rate=best_params["lr"]), loss="mse")
    autoencoder.summary()
    
    print("🟢 Compute reconstruction MSE")
    X_train_pred = autoencoder.predict(X_train)
    X_test_pred  = autoencoder.predict(X_test)

    mse_train = np.mean(np.square(X_train - X_train_pred), axis=1)
    mse_test  = np.mean(np.square(X_test - X_test_pred),  axis=1)

    print("Mean reconstruction MSE train:", np.mean(mse_train))
    print("Mean reconstruction MSE test:", np.mean(mse_test))

    print("🟢 Separar datos etiquetados de los no etiquetados")
    Z_train = encoder.predict(X_train)
    Z_test = encoder.predict(X_test)

    # Máscara para aislar la actividad no estructurada
    mask_unlabeled_train = (y_train == 'ACTIVIDAD NO ESTRUCTURADA') | (y_train == 'UNKNOWN')
    
    # 1. Conjunto inicial de entrenamiento (Solo etiquetas reales)
    Z_train_labeled = Z_train[~mask_unlabeled_train]
    y_train_labeled = y_train[~mask_unlabeled_train]
    
    # 2. La "bolsa" de datos sin etiquetar para el Self-Training
    Z_unlabeled = Z_train[mask_unlabeled_train]

    print("🟢 Build baseline classifier (Random Forest)")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(Z_train_labeled, y_train_labeled)

    print(f"🟢 Iniciar Self-Training Iterativo (Pool inicial: {len(Z_unlabeled)} muestras)")
    umbral = 0.90
    iteracion = 1

    # Bucle de Self-Training
    while len(Z_unlabeled) > 0:
        probas = clf.predict_proba(Z_unlabeled)
        max_probas = np.max(probas, axis=1)
        
        pred_classes = clf.classes_[np.argmax(probas, axis=1)]
        confident_mask = (max_probas >= umbral)
        nuevas_muestras = np.sum(confident_mask)
        
        # Salir si ninguna muestra supera el 90%
        if nuevas_muestras == 0:
            print(f"  -> Iteración {iteracion}: Ninguna muestra superó el {umbral*100}%. Fin del bucle.")
            break
            
        print(f"  -> Iteración {iteracion}: Añadidas {nuevas_muestras} muestras. Quedan {len(Z_unlabeled) - nuevas_muestras} en el pool.")
        
        # Juntar las muestras aprobadas al set de entrenamiento
        Z_train_labeled = np.vstack([Z_train_labeled, Z_unlabeled[confident_mask]])
        y_train_labeled = np.concatenate([y_train_labeled, pred_classes[confident_mask]])
        
        # Quitar las muestras aprobadas de la bolsa
        Z_unlabeled = Z_unlabeled[~confident_mask]
        
        # Reentrenar el clasificador
        clf.fit(Z_train_labeled, y_train_labeled)
        
        iteracion += 1

    print("🟢 Test classifier")
    # 3. Filtrado en el Test: Ocultamos lo "no estructurado" para evaluar solo aciertos reales
    mask_unlabeled_test = (y_test == 'ACTIVIDAD NO ESTRUCTURADA') | (y_test == 'UNKNOWN')
    Z_test_labeled = Z_test[~mask_unlabeled_test]
    y_test_labeled = y_test[~mask_unlabeled_test]

    y_test_pred = clf.predict(Z_test_labeled)
    acc_score_validation = accuracy_score(y_test_labeled, y_test_pred)
    f1_score_validation = f1_score(y_test_labeled, y_test_pred, average='macro')

    # save meta model metrics
    metric["loop"] = loop

    metric["classifier_model_accuracy"] = acc_score_validation
    metric["classifier_model_f1_score"] = f1_score_validation

    # add metrics to collection
    metrics.append(metric)

    elapsed_loop = time.perf_counter() - start_loop
    print(f"Loop time: {elapsed_loop:.2f} seconds")

print("🟢 Calculate metrics mean and standard deviations")
df_metrics = pd.DataFrame(metrics)

# Compute mean and std (numeric columns only)
mean_row = df_metrics.mean(numeric_only=True)
std_row = df_metrics.std(numeric_only=True)

# Add a label for the index column
mean_row["loop"] = "mean"
std_row["loop"] = "std"

# Append to dataframe
df_metrics = pd.concat(
    [df_metrics, pd.DataFrame([mean_row, std_row])],
    ignore_index=True
)

print("🟢 Save metrics")
# Apuntamos a la carpeta reports de tu árbol de proyecto
save_dir = Path.cwd() / "reports"
save_dir.mkdir(parents=True, exist_ok=True) # Crea la carpeta si no existiera

# Generamos el nombre dinámico con el sensor introducido
csv_filename = f"metrics_loocv_{args.sensor}.csv"
save_path = save_dir / csv_filename

df_metrics.to_csv(save_path, index=False)
print(f"Archivo guardado correctamente en: {save_path}")

elapsed_app = time.perf_counter() - start_app
print(f"Application time: {elapsed_app:.2f} seconds")