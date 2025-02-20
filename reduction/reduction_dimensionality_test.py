import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# 📌 Rutas
PREPROCESSED_PATH_TEST = r"C:\DAVID\CS\2025 0\machine_learning\P3_ML_GP6\pre_processing\test"
OUTPUT_PATH = r"C:\DAVID\CS\2025 0\machine_learning\P3_ML_GP6\reduced_data"

# Crear carpeta si no existe
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 📥 Cargar modelo PCA entrenado previamente
print("📥 Cargando modelo PCA entrenado...")
pca = IncrementalPCA(n_components=50)
pca.components_ = np.load(os.path.join(OUTPUT_PATH, "pca_components.npy"))
pca.mean_ = np.load(os.path.join(OUTPUT_PATH, "pca_mean.npy"))

# 📥 Cargar modelo LDA entrenado previamente
print("📥 Cargando modelo LDA entrenado...")
lda = LDA(n_components=6)
X_train_pca = np.load(os.path.join(OUTPUT_PATH, "pca_reduced.npy"))
y_train = np.load(os.path.join(OUTPUT_PATH, "y_train.npy"), allow_pickle=True)

# 🚨 Asegurar que `y_train` coincida con `X_train_pca`
min_samples = min(X_train_pca.shape[0], y_train.shape[0])
X_train_pca = X_train_pca[:min_samples]
y_train = y_train[:min_samples]

lda.fit(X_train_pca, y_train)

# 📥 Aplicar PCA y LDA a Test
print("⚙️ Aplicando PCA y LDA a Test...")

archivos_test = sorted([f for f in os.listdir(PREPROCESSED_PATH_TEST) if f.endswith(".npy")])
batch_size_pca = 100
X_reduced_pca = []
X_reduced_lda = []

for i in tqdm(range(0, len(archivos_test), batch_size_pca)):
    batch_files = archivos_test[i:i + batch_size_pca]
    X_batch = np.vstack([np.load(os.path.join(PREPROCESSED_PATH_TEST, file)).flatten() for file in batch_files])

    X_pca_batch = pca.transform(X_batch)
    X_lda_batch = lda.transform(X_pca_batch)

    X_reduced_pca.append(X_pca_batch)
    X_reduced_lda.append(X_lda_batch)

# 📥 Convertir listas a arrays numpy
X_reduced_pca = np.vstack(X_reduced_pca)
X_reduced_lda = np.vstack(X_reduced_lda)

# 💾 Guardar los archivos reducidos
np.save(os.path.join(OUTPUT_PATH, "pca_reduced_test.npy"), X_reduced_pca)
np.save(os.path.join(OUTPUT_PATH, "lda_reduced_test.npy"), X_reduced_lda)

print("✅ Reducción de dimensionalidad para Test completada.")
