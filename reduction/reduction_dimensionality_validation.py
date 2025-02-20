import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# 📌 Rutas
PREPROCESSED_PATH_TRAIN = r"C:\DAVID\CS\2025 0\machine_learning\Proyecto3\pequena_observacion_data\pre_processing"
PREPROCESSED_PATH_VALIDATION = r"C:\DAVID\CS\2025 0\machine_learning\P3_ML_GP6\pre_processing\validation"
OUTPUT_PATH = r"C:\DAVID\CS\2025 0\machine_learning\P3_ML_GP6\reduced_data"

# Crear carpeta si no existe
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 📥 Inicializar IncrementalPCA
batch_size_pca = 100
pca = IncrementalPCA(n_components=50)

# 📥 Entrenar PCA en `train` en lotes
print("⚙️ Entrenando IncrementalPCA por lotes con imágenes de TRAIN...")
archivos_train = sorted([f for f in os.listdir(PREPROCESSED_PATH_TRAIN) if f.endswith(".npy")])

for i in tqdm(range(0, len(archivos_train), batch_size_pca)):
    batch_files = archivos_train[i:i + batch_size_pca]

    if len(batch_files) < 50:  
        print(f"⚠️ Omitiendo último lote con {len(batch_files)} imágenes (menor que n_components=50).")
        continue

    X_batch = np.vstack([np.load(os.path.join(PREPROCESSED_PATH_TRAIN, file)).flatten() for file in batch_files])
    pca.partial_fit(X_batch)

# 💾 Guardar modelo PCA entrenado
np.save(os.path.join(OUTPUT_PATH, "pca_components.npy"), pca.components_)
np.save(os.path.join(OUTPUT_PATH, "pca_mean.npy"), pca.mean_)

# 📥 Aplicar PCA a TRAIN y entrenar LDA
lda = LDA(n_components=6)
X_train_pca = []

print("⚙️ Aplicando PCA a TRAIN para entrenar LDA...")
for i in tqdm(range(0, len(archivos_train), batch_size_pca)):
    batch_files = archivos_train[i:i + batch_size_pca]
    X_batch = np.vstack([np.load(os.path.join(PREPROCESSED_PATH_TRAIN, file)).flatten() for file in batch_files])
    X_train_pca.append(pca.transform(X_batch))

X_train_pca = np.vstack(X_train_pca)

# 📥 Cargar etiquetas y asegurar que coincidan en tamaño con `X_train_pca`
y_train = np.load(os.path.join(OUTPUT_PATH, "y_train.npy"), allow_pickle=True)

# 🚨 Ajustar `y_train` al tamaño de `X_train_pca` si es necesario
if X_train_pca.shape[0] != y_train.shape[0]:
    min_samples = min(X_train_pca.shape[0], y_train.shape[0])
    X_train_pca = X_train_pca[:min_samples]
    y_train = y_train[:min_samples]

lda.fit(X_train_pca, y_train)

# 📥 Aplicar PCA y LDA a Validation
print("⚙️ Aplicando PCA y LDA a Validation...")

archivos_val = sorted([f for f in os.listdir(PREPROCESSED_PATH_VALIDATION) if f.endswith(".npy")])
X_reduced_pca = []
X_reduced_lda = []

for i in tqdm(range(0, len(archivos_val), batch_size_pca)):
    batch_files = archivos_val[i:i + batch_size_pca]
    X_batch = np.vstack([np.load(os.path.join(PREPROCESSED_PATH_VALIDATION, file)).flatten() for file in batch_files])

    X_pca_batch = pca.transform(X_batch)
    X_lda_batch = lda.transform(X_pca_batch)

    X_reduced_pca.append(X_pca_batch)
    X_reduced_lda.append(X_lda_batch)

# 📥 Convertir listas a arrays numpy
X_reduced_pca = np.vstack(X_reduced_pca)
X_reduced_lda = np.vstack(X_reduced_lda)

# 💾 Guardar los archivos reducidos
np.save(os.path.join(OUTPUT_PATH, "pca_reduced_validation.npy"), X_reduced_pca)
np.save(os.path.join(OUTPUT_PATH, "lda_reduced_validation.npy"), X_reduced_lda)

print("✅ Reducción de dimensionalidad para Validation completada.")
