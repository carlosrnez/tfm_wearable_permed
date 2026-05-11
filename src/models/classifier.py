import torch
import torch.nn as nn

class WearableClassifier(nn.Module):
    """
    Fase 1.1: Clasificador base.
    Utiliza un Encoder pre-entrenado (Fase 0) como extractor de características
    y le añade una capa densa final (clasificador) con una neurona por clase.
    """
    def __init__(self, encoder, latent_dim, num_clases, congelar_encoder=False):
        super(WearableClassifier, self).__init__()
        
        # 1. El Encoder pre-entrenado (nuestro extractor de características de 91 -> 16)
        self.encoder = encoder
        
        # Opción para "congelar" los pesos del encoder durante las primeras épocas
        # Si está True, el entrenamiento solo ajustará la capa final.
        if congelar_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        # 2. La capa de clasificación (16 -> Número de actividades)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, num_clases)
            # No aplicamos Softmax aquí porque PyTorch incluye la función Softmax 
            # internamente en su función de pérdida (CrossEntropyLoss)
        )

    def forward(self, x):
        # 1. Pasamos los datos por el encoder pre-entrenado para comprimirlos
        # x_latente tendrá dimensiones (Batch_Size, 16)
        x_latente = self.encoder(x)
        
        # 2. Pasamos la compresión por la capa final para predecir la actividad
        # salida tendrá dimensiones (Batch_Size, num_clases)
        salida = self.classifier(x_latente)
        
        return salida

if __name__ == "__main__":
    # --- PRUEBA RÁPIDA DE DIMENSIONES (DUMMY TEST) ---
    print("Probando dimensiones del Clasificador...")
    
    # Importamos el Autoencoder solo para la prueba
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from models.autoencoder import DenseAutoencoder
    
    # Simulamos el entorno
    batch_size = 32
    input_features = 91
    latent_dim = 16
    num_clases_simuladas = 16
    
    # Creamos un encoder vacío (simulando que viene de la Fase 0)
    autoencoder = DenseAutoencoder(input_dim=input_features, latent_dim=latent_dim)
    encoder_solo = autoencoder.get_encoder()
    
    # Instanciamos el clasificador
    modelo = WearableClassifier(encoder_solo, latent_dim, num_clases_simuladas)
    
    # Probamos con datos falsos
    datos_prueba = torch.randn(batch_size, input_features)
    salida = modelo(datos_prueba)
    
    print(f"Entrada (Datos brutos): {datos_prueba.shape}")
    print(f"Salida (Predicciones por clase): {salida.shape}")
    
    if salida.shape == (batch_size, num_clases_simuladas):
        print("\n✅ ¡ÉXITO! El clasificador engarza perfectamente con el Encoder y las dimensiones cuadran.")
    else:
        print("\n❌ ERROR: Las dimensiones de salida no son las esperadas.")