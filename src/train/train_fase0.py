import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Esto permite que Python encuentre las carpetas 'models' y 'data'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.autoencoder import DenseAutoencoder
from data.DataReader import preparar_datos_tfm

def entrenar_fase0(ruta_datos, epochs=5, batch_size=256, lr=1e-3):
    print("Iniciando preparación de datos...")
    loader_L, loader_U, _ = preparar_datos_tfm(ruta_datos, batch_size=batch_size)
    
    # 1. Configuración de dispositivo (Usa tarjeta gráfica si la tienes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDispositivo de entrenamiento: {device}")
    
    # 2. Inicializar Modelo, Función de Pérdida y Optimizador
    model = DenseAutoencoder(input_dim=91, latent_dim=16).to(device)
    criterion = nn.MSELoss() # Error Cuadrático Medio: penaliza si la reconstrucción es distinta a la entrada
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("\n--- INICIANDO ENTRENAMIENTO FASE 0 (AUTOENCODER) ---")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        batches = 0
        
        # A. Entrenamos con los datos etiquetados (Set L)
        for batch in loader_L:
            X, _ = batch           # Desempaquetamos e ignoramos las etiquetas
            X = X.to(device)
            
            optimizer.zero_grad()
            reconstruccion = model(X)
            loss = criterion(reconstruccion, X)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            batches += 1
            
        # B. Entrenamos con los datos masivos sin etiquetar (Set U)
        for X in loader_U:
            X = X.to(device)       # Aquí no hay etiquetas que ignorar
            
            optimizer.zero_grad()
            reconstruccion = model(X)
            loss = criterion(reconstruccion, X)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            batches += 1
            
        # Calculamos la pérdida media de la época
        avg_loss = train_loss / batches
        print(f"Época [{epoch+1}/{epochs}] - Loss Reconstrucción: {avg_loss:.4f}")
        
    # 3. Guardar el modelo pre-entrenado
    os.makedirs("../../models_saved", exist_ok=True)
    ruta_guardado = "../../models_saved/autoencoder_fase0.pth"
    torch.save(model.state_dict(), ruta_guardado)
    print(f"\n✅ ¡Fase 0 completada! Modelo guardado en '{ruta_guardado}'")

if __name__ == "__main__":
    # Sustituye por tu ruta exacta
    ruta = "C:/Users/CARLOS/Desktop/dev/tfm_wearablepermed/data/processed/sensor_M/data_feature_all_M.npz"
    entrenar_fase0(ruta, epochs=5) # Empezamos con 5 épocas para ver si baja el loss