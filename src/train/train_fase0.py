import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import time
import csv
from datetime import datetime

# Detectar la raíz del proyecto
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from models.autoencoder import DenseAutoencoder
from data.DataReader import preparar_datos_tfm

def guardar_metricas(exp_name, sensor, epochs, batch_size, lr, metricas, tiempo_segundos):
    """Guarda las métricas avanzadas en el CSV maestro."""
    carpeta_reports = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(carpeta_reports, exist_ok=True)
    
    ruta_csv = os.path.join(carpeta_reports, "fase0_metricas.csv")
    archivo_existe = os.path.isfile(ruta_csv)
    
    with open(ruta_csv, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not archivo_existe:
            writer.writerow([
                'Fecha', 'Experimento', 'Sensor', 'Epochs', 'Batch_Size', 'LR', 
                'MSE_L_Inicial', 'MSE_U_Inicial', 'MSE_L_Final', 'MSE_U_Final', 
                'MAE_L_Final', 'MAE_U_Final', 'Tiempo_Segundos'
            ])
        
        fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([
            fecha_actual, exp_name, sensor, epochs, batch_size, lr, 
            f"{metricas['mse_l_ini']:.4f}", f"{metricas['mse_u_ini']:.4f}", 
            f"{metricas['mse_l_fin']:.4f}", f"{metricas['mse_u_fin']:.4f}", 
            f"{metricas['mae_l_fin']:.4f}", f"{metricas['mae_u_fin']:.4f}", 
            f"{tiempo_segundos:.2f}"
        ])

def entrenar_fase0(ruta_datos, sensor, exp_name, epochs, batch_size, lr):
    print(f"Iniciando Experimento: '{exp_name}' | Sensor: '{sensor}'")
    
    loader_L, loader_U, _ = preparar_datos_tfm(ruta_datos, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDispositivo de entrenamiento: {device}")
    
    model = DenseAutoencoder(input_dim=91, latent_dim=16).to(device)
    
    # Definimos ambas funciones de pérdida (MSE para entrenar, MAE para métricas extra)
    criterion_mse = nn.MSELoss() 
    criterion_mae = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    metricas = {}
    start_time = time.time()
    
    print("\n--- INICIANDO ENTRENAMIENTO FASE 0 (AUTOENCODER) ---")
    for epoch in range(epochs):
        model.train()
        
        mse_l, mae_l, batches_l = 0.0, 0.0, 0
        mse_u, mae_u, batches_u = 0.0, 0.0, 0
        
        # A. Entrenamiento Set L (Etiquetados)
        for batch in loader_L:
            X, _ = batch           
            X = X.to(device)
            
            optimizer.zero_grad()
            reconstruccion = model(X)
            
            loss = criterion_mse(reconstruccion, X)
            loss.backward()
            optimizer.step()
            
            mse_l += loss.item()
            # Calculamos MAE solo como métrica (sin .backward)
            with torch.no_grad():
                mae_l += criterion_mae(reconstruccion, X).item()
            batches_l += 1
            
        # B. Entrenamiento Set U (No etiquetados)
        for X in loader_U:
            X = X.to(device)       
            
            optimizer.zero_grad()
            reconstruccion = model(X)
            
            loss = criterion_mse(reconstruccion, X)
            loss.backward()
            optimizer.step()
            
            mse_u += loss.item()
            with torch.no_grad():
                mae_u += criterion_mae(reconstruccion, X).item()
            batches_u += 1
            
        # Promedios de la época
        avg_mse_l = mse_l / batches_l if batches_l > 0 else 0
        avg_mse_u = mse_u / batches_u if batches_u > 0 else 0
        avg_mae_l = mae_l / batches_l if batches_l > 0 else 0
        avg_mae_u = mae_u / batches_u if batches_u > 0 else 0
        
        # Mostrar progreso unificado por terminal
        print(f"Época [{epoch+1}/{epochs}] | MSE Set L: {avg_mse_l:.2f} | MSE Set U: {avg_mse_u:.2f} | MAE Set U: {avg_mae_u:.2f}")
        
        # Registrar iniciales y finales para el CSV
        if epoch == 0:
            metricas['mse_l_ini'] = avg_mse_l
            metricas['mse_u_ini'] = avg_mse_u
        if epoch == epochs - 1:
            metricas['mse_l_fin'] = avg_mse_l
            metricas['mse_u_fin'] = avg_mse_u
            metricas['mae_l_fin'] = avg_mae_l
            metricas['mae_u_fin'] = avg_mae_u
            
    tiempo_total = time.time() - start_time
        
    # Guardado del modelo
    carpeta_modelos = os.path.join(PROJECT_ROOT, "models_saved", "fase0", f"sensor_{sensor}")
    os.makedirs(carpeta_modelos, exist_ok=True)
    ruta_guardado = os.path.join(carpeta_modelos, f"autoencoder_{exp_name}.pth")
    torch.save(model.state_dict(), ruta_guardado)
    
    # Guardado de métricas
    guardar_metricas(exp_name, sensor, epochs, batch_size, lr, metricas, tiempo_total)
                     
    print("\n✅ ¡Entrenamiento completado!")
    print(f"🧠 Modelo guardado en: {ruta_guardado}")
    print(f"📊 Métricas guardadas en: {os.path.join(PROJECT_ROOT, 'reports', 'fase0_metricas.csv')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de entrenamiento Fase 0")
    parser.add_argument("--ruta_datos", type=str, required=True, help="Ruta al archivo .npz")
    parser.add_argument("--sensor", type=str, required=True, choices=['M', 'PI'], help="Sensor: M o PI")
    parser.add_argument("--exp_name", type=str, required=True, help="Nombre del experimento")
    parser.add_argument("--epochs", type=int, default=5, help="Número de épocas")
    parser.add_argument("--batch_size", type=int, default=256, help="Tamaño del batch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()
    
    entrenar_fase0(args.ruta_datos, args.sensor, args.exp_name, args.epochs, args.batch_size, args.lr)