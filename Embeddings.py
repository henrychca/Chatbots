# ==============================================
# IMPORTACIÓN DE LIBRERÍAS
# ==============================================
# SentenceTransformer: para generar embeddings semánticos de texto.
# numpy: para operaciones numéricas.
# PCA: reducción lineal de dimensionalidad a 2D (rápida).
# TSNE: reducción no lineal para visualizaciones más orgánicas.
# matplotlib: para graficar los embeddings.
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# ==============================================
# CARGA DEL MODELO DE EMBEDDINGS
# ==============================================
# Se carga un modelo preentrenado ("thenlper/gte-small"),
# que convierte oraciones o palabras en vectores numéricos
# que reflejan su significado semántico.
model = SentenceTransformer("thenlper/gte-small")
print("Modelo de embeddings 'gte-small' cargado correctamente.\n")

# ==============================================
# DICCIONARIO BASE DE TEXTOS RELACIONADOS CON EL HOGAR
# ==============================================
# Lista de frases base que servirán como referencia o "ancla"
# para comparar con los textos que el usuario escriba.
textos_hogar = [
    "casa", "hogar", "habitación", "sala", "comedor",
    "el gato duerme en el sofá", "la comida está en la mesa",
    "la luz del baño está encendida", "abro la puerta del cuarto"
]

# Genera los embeddings del diccionario base.
embeddings_base = model.encode(textos_hogar)

# Estas listas vacías guardarán los textos y embeddings
# que el usuario ingrese durante la sesión interactiva.
textos_usuario = []
embeddings_usuario = []

# ==============================================
# BUCLE INTERACTIVO DE ENTRADA DEL USUARIO
# ==============================================
print("Escribe una palabra o frase relacionada con el hogar.")
print("Escribe 'salir' o 'exit' para finalizar.\n")

while True:
    # Solicita un texto al usuario
    user_input = input("Ingresa texto: ").strip()

    # Si escribe 'salir' o 'exit', se interrumpe el bucle
    if user_input.lower() in ["salir", "exit"]:
        print("\nGenerando visualización final...")
        break

    # Genera el embedding para el texto ingresado
    emb = model.encode([user_input])[0]

    # Calcula la norma (longitud) del vector como referencia
    norma = np.linalg.norm(emb)

    # Guarda el texto y su embedding
    textos_usuario.append(user_input)
    embeddings_usuario.append(emb)

    # Muestra un resumen del vector generado
    print(f"\nEmbedding generado para: {user_input}")
    print(f"Dimensión: {len(emb)} | Norma: {norma:.4f}")
    print(f"Primeros 10 valores: {np.round(emb[:10], 4)}\n")

# ==============================================
# COMBINACIÓN DE EMBEDDINGS BASE Y DEL USUARIO
# ==============================================
# Si el usuario ingresó textos, los unimos con los embeddings base.
if embeddings_usuario:
    textos_total = textos_hogar + textos_usuario
    embeddings_total = np.vstack([embeddings_base, np.vstack(embeddings_usuario)])
else:
    # Si no escribió nada, solo se grafican los textos base
    textos_total = textos_hogar
    embeddings_total = embeddings_base

# ==============================================
# SELECCIÓN DEL MÉTODO DE REDUCCIÓN (PCA o t-SNE)
# ==============================================
print("\nMétodos de reducción disponibles:")
print("1. PCA  (más rápido, lineal, útil para estructuras globales)")
print("2. t-SNE (más visual, no lineal, revela agrupaciones semánticas)")
opcion = input("Selecciona método [1=PCA | 2=t-SNE]: ").strip()

# ==============================================
# REDUCCIÓN DE DIMENSIONALIDAD
# ==============================================
if opcion == "2":
    # ----------------------------------------------
    # t-SNE: técnica no lineal (más lenta pero expresiva)
    # ----------------------------------------------
    print("\nAplicando t-SNE (esto puede tardar unos segundos)...")

    # t-SNE tiende a mostrar agrupamientos más orgánicos
    # pero puede variar entre ejecuciones. Usa un random_state
    # para hacerlo reproducible.
    # En versiones nuevas de scikit-learn se usa 'max_iter' en lugar de 'n_iter'.
    try:
        tsne = TSNE(
            n_components=2,
            perplexity=5,      # Cuántos vecinos considera (ajustable según cantidad de puntos)
            learning_rate=200, # Tamaño del paso en el proceso de optimización
            max_iter=1000,     # Iteraciones de entrenamiento (parámetro actualizado)
            random_state=42,
            init="random"
        )
    except TypeError:
        # Compatibilidad con versiones anteriores
        tsne = TSNE(
            n_components=2,
            perplexity=5,
            learning_rate=200,
            n_iter=1000,
            random_state=42,
            init="random"
        )

    coords = tsne.fit_transform(embeddings_total)
    metodo = "t-SNE"

else:
    # ----------------------------------------------
    # PCA: técnica lineal (más rápida)
    # ----------------------------------------------
    print("\nAplicando PCA para reducción a 2 dimensiones...")
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings_total)
    metodo = "PCA"

# ==============================================
# SEPARACIÓN DE COORDENADAS BASE Y USUARIO
# ==============================================
n_base = len(embeddings_base)
coords_base = coords[:n_base]      # Coordenadas del diccionario base
coords_usr = coords[n_base:]       # Coordenadas de textos del usuario

# ==============================================
# VISUALIZACIÓN GRÁFICA CON MATPLOTLIB
# ==============================================
plt.figure(figsize=(10, 7))
plt.title(f"Visualización de Embeddings ({metodo})", fontsize=14, color="navy")

# Dibuja los puntos del diccionario base (azul)
plt.scatter(coords_base[:, 0], coords_base[:, 1],
            c="skyblue", label="Diccionario base", s=80)

# Dibuja los puntos del usuario (rojo con estrella)
if embeddings_usuario:
    plt.scatter(coords_usr[:, 0], coords_usr[:, 1],
                c="tomato", label="Ingresados por usuario", s=100, marker="*")

# Añade etiquetas de texto a cada punto
for i, txt in enumerate(textos_total):
    plt.annotate(txt, (coords[i, 0] + 0.02, coords[i, 1] + 0.02), fontsize=9)

# Personalización del gráfico
plt.legend()
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

print(f"\nVisualización completada con {metodo}.")
