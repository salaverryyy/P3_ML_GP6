import os
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Rutas
PREPROCESSED_PATH = r"C:\DAVID\CS\2025 0\machine_learning\Proyecto3\P3_ML_GP6\pre_processing"
LABELS_FILE = r"C:\DAVID\CS\2025 0\machine_learning\Data_Proyect3\ISIC2018_Task3_Training_GroundTruth\ISIC2018_Task3_Training_GroundTruth\ISIC2018_Task3_Training_GroundTruth.csv"
OUTPUT_PATH = r"C:\DAVID\CS\2025 0\machine_learning\Proyecto3\P3_ML_GP6\reduced_data"

# Crear carpeta si no existe
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("📥 Cargando etiquetas...")
df = pd.read_csv(LABELS_FILE, sep=';')
df['diagnosis'] = df.iloc[:, 1:].idxmax(axis=1)

# Configurar tamaño de lotes
batch_size_pca = 250  # Reducimos el tamaño del lote para PCA
batch_size_lda = 100  # Más pequeño para LDA
num_batches_pca = len(df) // batch_size_pca + 1
num_batches_lda = len(df) // batch_size_lda + 1

# Definir IncrementalPCA y LDA
pca = IncrementalPCA(n_components=50, batch_size=batch_size_pca)
lda = LDA(n_components=6)

print("⚙️ Aplicando PCA en lotes con `IncrementalPCA`...")

# 🔹 1. Entrenar PCA en lotes pequeños sin sobrecargar la RAM
pca_trained = False  # Para verificar si al menos un lote fue entrenado

for i in range(num_batches_pca):
    print(f"📦 Procesando PCA - Lote {i+1}/{num_batches_pca}...")

    X_batch = []
    y_batch = []

    for _, row in tqdm(df.iloc[i * batch_size_pca: (i + 1) * batch_size_pca].iterrows(), total=batch_size_pca):
        image_name = row['image'] + ".npy"
        image_path = os.path.join(PREPROCESSED_PATH, image_name)

        if os.path.exists(image_path):
            img = np.load(image_path).flatten()
            X_batch.append(img)
            y_batch.append(row['diagnosis'])

    if len(X_batch) == 0:
        continue

    X_batch = np.array(X_batch, dtype=np.float16)
    y_batch = np.array(y_batch)

    if len(X_batch) >= pca.n_components:
        print("🔹 Ajustando PCA con este lote...")
        pca.partial_fit(X_batch)  # Usa `partial_fit()` solo si el lote es suficientemente grande
        pca_trained = True
    else:
        print(f"⚠️ Omitiendo el último lote ({len(X_batch)} muestras) porque es menor que n_components={pca.n_components}")

# Solo guardar los componentes si al menos un lote se procesó
if pca_trained:
    print("✅ PCA entrenado, guardando componentes...")
    np.save(os.path.join(OUTPUT_PATH, "pca_components.npy"), pca.components_)
else:
    print("❌ No se pudo entrenar PCA debido a lotes demasiado pequeños.")

# 🔹 Aplicar transformación PCA en lotes
print("⚙️ Aplicando transformación PCA...")
for i in range(num_batches_pca):
    print(f"📦 Transformando PCA - Lote {i+1}/{num_batches_pca}...")

    X_batch = []
    for _, row in tqdm(df.iloc[i * batch_size_pca: (i + 1) * batch_size_pca].iterrows(), total=batch_size_pca):
        image_name = row['image'] + ".npy"
        image_path = os.path.join(PREPROCESSED_PATH, image_name)

        if os.path.exists(image_path):
            img = np.load(image_path).flatten()
            X_batch.append(img)

    if len(X_batch) == 0:
        continue

    X_batch = np.array(X_batch, dtype=np.float16)

    if len(X_batch) >= pca.n_components:
        print("🔹 Aplicando transformación PCA...")
        X_pca_batch = pca.transform(X_batch)
        np.save(os.path.join(OUTPUT_PATH, f"pca_reduced_batch_{i}.npy"), X_pca_batch)
        print(f"✅ Lote {i+1}/{num_batches_pca} procesado y guardado para PCA.")
    else:
        print(f"⚠️ Omitiendo la transformación de PCA para lote {i+1} porque tiene solo {len(X_batch)} muestras.")

# 🔹 2. Procesar LDA en lotes más pequeños
print("⚙️ Aplicando LDA en lotes...")
first_batch_lda = True
for i in range(num_batches_lda):
    print(f"📦 Procesando LDA - Lote {i+1}/{num_batches_lda}...")

    X_batch = []
    y_batch = []

    for _, row in tqdm(df.iloc[i * batch_size_lda: (i + 1) * batch_size_lda].iterrows(), total=batch_size_lda):
        image_name = row['image'] + ".npy"
        image_path = os.path.join(PREPROCESSED_PATH, image_name)

        if os.path.exists(image_path):
            img = np.load(image_path).flatten()
            X_batch.append(img)
            y_batch.append(row['diagnosis'])

    if len(X_batch) == 0:
        continue

    X_batch = np.array(X_batch, dtype=np.float16)
    y_batch = np.array(y_batch)

    if first_batch_lda and len(X_batch) >= lda.n_components:
        print("🔹 Entrenando LDA con primer lote...")
        lda.fit(X_batch, y_batch)
        first_batch_lda = False

    if len(X_batch) >= lda.n_components:
        print("🔹 Aplicando transformación LDA...")
        X_lda_batch = lda.transform(X_batch)
        np.save(os.path.join(OUTPUT_PATH, f"lda_reduced_batch_{i}.npy"), X_lda_batch)
        print(f"✅ Lote {i+1}/{num_batches_lda} procesado y guardado para LDA.")
    else:
        print(f"⚠️ Omitiendo la transformación de LDA para lote {i+1} porque tiene solo {len(X_batch)} muestras.")

print("🎉 Reducción de dimensionalidad completada.")
