# TFM: Reconocimiento Actividad Física Semisupervisado (WearablePerMed)

## Objetivo
Clasificación de actividad física integrando características de sensores de muñeca y muslo usando aprendizaje semisupervisado para aprovechar datos no etiquetados en "vida libre".

## Formato de los Datos
- **Tipo de red:** Dense Autoencoder (Multicapa Perceptrón)
- **Dimensiones de entrada:** 91 características tabulares (estadísticas pre-extraídas por ventana).
- **Etiquetas (`arr_1`):** 1 clase masiva de "ACTIVIDAD NO ESTRUCTURADA" (~570k muestras) y varias clases etiquetadas de laboratorio (~3k muestras).
- **Sujetos (`arr_2`):** 5 pacientes identificados (ej. PMP1025). 

## Partición (Leave-One-Subject-Out)
- **Set T (Test):** Todas las ventanas etiquetadas de **un sujeto excluido** (ej. PMP1025). Aislado hasta la evaluación final.
- **Set L (Labeled):** Ventanas etiquetadas de los 4 sujetos restantes.
- **Set U (Unlabeled):** Ventanas de la clase "ACTIVIDAD NO ESTRUCTURADA" de los 4 sujetos de entrenamiento.

## Hoja de Ruta
- [x] **Arquitectura Autoencoder:** DenseAutoencoder creado y validado (`input_dim=91`).
- [ ] **Data Loader:** Partición estricta L, U, T sin fugas de datos.
- [ ] **Fase 0:** Entrenamiento del Autoencoder con L + U.
- [ ] **Fase 1:** Baseline Supervisado y Self-Training.
- [ ] **Fase 2:** Adaptabilidad (SAT) con umbrales dinámicos.
- [ ] **Fase 3:** FixMatch.