# 📊 Diario de Experimentos - WearablePerMed

Este documento registra cualitativamente los avances, hiperparámetros y conclusiones de cada fase del modelo de reconocimiento de actividad física (aprendizaje semisupervisado).

---

## Fase 0: Aprendizaje de Representaciones (Autoencoder)
**Objetivo:** Comprimir las 91 características de los sensores en un espacio latente de 16 dimensiones utilizando datos etiquetados (Set L) y no etiquetados (Set U).

### Experimento 0.1: Sensor Muñeca (M)
* **Fecha:** 06/05/2026
* **Arquitectura:** DenseAutoencoder (Input: 91 -> Latent: 16)
* **Hiperparámetros:** 
  * Epochs: 5 
  * Batch Size: 256 
  * Learning Rate: 0.001 (1e-3)
* **Resultados de Entrenamiento:** 
  * Loss Inicial (Época 1): 9478.41
  * Loss Final (Época 5): 1898.27
* **Conclusiones y Siguientes Pasos:** 
  * ✅ **Éxito:** El modelo aprende y converge correctamente; la *loss* se reduce drásticamente.
  * ⚠️ **Nota:** Los valores absolutos del error son muy altos. Esto indica casi con total seguridad que las 91 características originales no están normalizadas (Z-score). Para las siguientes fases, es vital aplicar una normalización (`StandardScaler`) antes de pasar los datos a la red neuronal.

### Experimento 0.2: Sensor Muslo (PI)
* [Pendiente de ejecutar con el archivo .npz de la pierna]

---

## Fase 1: Baseline Supervisado y Self-Training
* [Pendiente]