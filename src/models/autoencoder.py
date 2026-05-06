import torch
import torch.nn as nn

class DenseAutoencoder(nn.Module):
    def __init__(self, input_dim=91, latent_dim=16):
        """
        Autoencoder basado en capas densas para datos tabulares (características pre-extraídas).
        - input_dim: 91 (número de características de nuestro dataset).
        - latent_dim: Tamaño del cuello de botella (espacio latente comprimido).
        """
        super(DenseAutoencoder, self).__init__()
        
        # ENCODER: Comprime los 91 datos originales hasta el espacio latente
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Linear(32, latent_dim),
            # No ponemos ReLU al final del latente para no restringir los valores a positivos
        )
        
        # DECODER: Intenta reconstruir los 91 datos a partir de la representación comprimida
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, input_dim)
            # No aplicamos activación final asumiendo que las características de entrada 
            # están estandarizadas (Z-score normalizadas) y pueden tener valores negativos.
        )

    def forward(self, x):
        """
        Flujo normal: comprime y luego reconstruye (usado para la Fase 0).
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def get_encoder(self):
        """
        Devuelve solo el bloque Encoder. Vital para la Fase 1 (Baseline Supervisado),
        donde le conectaremos una capa de clasificación final.
        """
        return self.encoder

if __name__ == "__main__":
    # --- PRUEBA RÁPIDA (DUMMY TEST) ---
    print("Iniciando prueba del Autoencoder...")
    
    # 1. Instanciamos el modelo
    modelo = DenseAutoencoder(input_dim=91, latent_dim=16)
    
    # 2. Creamos un lote de datos falsos (Batch size: 32 ventanas, Features: 91)
    # torch.randn genera números aleatorios simulando nuestros datos normalizados
    datos_prueba = torch.randn(32, 91)
    
    # 3. Pasamos los datos por el modelo (hacemos un 'forward pass')
    salida = modelo(datos_prueba)
    
    # 4. Comprobamos que el Encoder funciona aislado
    encoder_solo = modelo.get_encoder()
    latente = encoder_solo(datos_prueba)
    
    # 5. Imprimimos resultados
    print(f"Dimensiones de entrada: {datos_prueba.shape}")
    print(f"Dimensiones latentes (cuello de botella): {latente.shape}")
    print(f"Dimensiones de salida reconstruida: {salida.shape}")
    
    if datos_prueba.shape == salida.shape:
        print("\n✅ ¡ÉXITO! El modelo compila y las dimensiones cuadran a la perfección.")
    else:
        print("\n❌ ERROR: Las dimensiones no coinciden.")