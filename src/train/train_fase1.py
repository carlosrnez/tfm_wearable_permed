import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import time
import csv
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# Configuración de rutas absolutas
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from models.autoencoder import DenseAutoencoder
from models.classifier import WearableClassifier
from data.DataReader import preparar_datos_tfm

def guardar_metricas_f1(exp_name, sensor, metrics_train, metrics_test, loss_f, tiempo_segundos):
    """Guarda un registro exhaustivo de métricas en el CSV maestro."""
    carpeta_reports = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(carpeta_reports, exist_ok=True)
    ruta_csv = os.path.join(carpeta_reports, "fase1_metricas.csv")
    archivo_existe = os.path.isfile(ruta_csv)
    
    with open(ruta_csv, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not archivo_existe:
            writer.writerow([
                'Fecha', 'Exp', 'Sensor', 'Loss_Train', 
                'Acc_Train', 'Acc_Test', 
                'Prec_Macro_Test', 'Recall_Macro_Test', 'F1_Macro_Test', 
                'Tiempo_Seg'
            ])
        
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M"), 
            exp_name, sensor, f"{loss_f:.4f}", 
            f"{metrics_train['acc']:.4f}", f"{metrics_test['acc']:.4f}", 
            f"{metrics_test['precision']:.4f}", f"{metrics_test['recall']:.4f}", f"{metrics_test['f1']:.4f}", 
            f"{tiempo_segundos:.2f}"
        ])

def evaluar_modelo(model, loader, device):
    """Calcula métricas avanzadas (Acc, Precision, Recall, F1) para un conjunto de datos."""
    model.eval()
    todas_preds = []
    todas_labels = []
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, preds = torch.max(outputs, 1)
            todas_preds.extend(preds.cpu().numpy())
            todas_labels.extend(y.cpu().numpy())
            
    # Calculamos métricas con average='macro' para tratar todas las clases por igual
    acc = accuracy_score(todas_labels, todas_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        todas_labels, todas_preds, average='macro', zero_division=0
    )
    
    return {'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1}, todas_labels, todas_preds

def entrenar_fase1(ruta_datos, ruta_modelo_f0, sensor, exp_name, epochs, lr):
    print(f"\n--- INICIANDO FASE 1.1 (SUPERVISADO) | SENSOR: {sensor} | EXP: {exp_name} ---")
    
    # 1. Carga de datos
    loader_L, _, loader_T, n_clases = preparar_datos_tfm(ruta_datos)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    
    # 2. Inyección del Encoder pre-entrenado
    print(f"Cargando pesos de Fase 0 desde: {ruta_modelo_f0}")
    ae_previo = DenseAutoencoder(input_dim=91, latent_dim=16)
    try:
        ae_previo.load_state_dict(torch.load(ruta_modelo_f0, map_location=device))
    except Exception as e:
        print(f"\n❌ ERROR: No se pudo cargar el modelo de Fase 0. Comprueba la ruta.\nDetalles: {e}")
        return

    encoder_f0 = ae_previo.get_encoder()
    
    # 3. Preparación del Clasificador
    model = WearableClassifier(encoder_f0, latent_dim=16, num_clases=n_clases).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    start_time = time.time()
    
    # 4. Bucle de Entrenamiento
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for X, y in loader_L:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_loss = train_loss / len(loader_L)
        
        # Evaluar en cada época
        metrics_train, _, _ = evaluar_modelo(model, loader_L, device)
        metrics_test, _, _ = evaluar_modelo(model, loader_T, device)
        
        print(f"Época [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | "
              f"Acc Train: {metrics_train['acc']:.2%} | Acc Test: {metrics_test['acc']:.2%} | F1 Test: {metrics_test['f1']:.4f}")

    # 5. Evaluación Final Detallada (Console Report)
    print("\n" + "="*50)
    print("📊 REPORTE DE CLASIFICACIÓN FINAL (SET TEST - SUJETO AISLADO)")
    print("="*50)
    _, labels_finales, preds_finales = evaluar_modelo(model, loader_T, device)
    # Mostramos el reporte completo por consola para análisis cualitativo
    print(classification_report(labels_finales, preds_finales, zero_division=0))
    
    # 6. Guardado de Modelo y Métricas
    tiempo_total = time.time() - start_time
    carpeta_modelos = os.path.join(PROJECT_ROOT, "models_saved", "fase1", f"sensor_{sensor}")
    os.makedirs(carpeta_modelos, exist_ok=True)
    ruta_final = os.path.join(carpeta_modelos, f"classifier_{exp_name}.pth")
    torch.save(model.state_dict(), ruta_final)
    
    guardar_metricas_f1(exp_name, sensor, metrics_train, metrics_test, avg_loss, tiempo_total)
    
    print(f"\n✅ Fase 1.1 completada.")
    print(f"🧠 Modelo guardado en: {ruta_final}")
    print(f"📈 Métricas maestras guardadas en: reports/fase1_metricas.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fase 1.1: Entrenamiento Supervisado del Clasificador")
    parser.add_argument("--ruta_datos", type=str, required=True, help="Ruta al archivo .npz")
    parser.add_argument("--ruta_f0", type=str, required=True, help="Ruta al .pth del Autoencoder")
    # ✅ Cambio implementado: 'M' (Muñeca) y 'PI' (Pierna)
    parser.add_argument("--sensor", type=str, choices=['M', 'PI'], required=True, help="Sensor: M o PI")
    parser.add_argument("--exp_name", type=str, required=True, help="Nombre del experimento")
    parser.add_argument("--epochs", type=int, default=20, help="Épocas (default: 20)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate (default: 0.001)")
    
    args = parser.parse_args()
    entrenar_fase1(args.ruta_datos, args.ruta_f0, args.sensor, args.exp_name, args.epochs, args.lr)