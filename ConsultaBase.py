# ======================================================================
# CONSULTA DE BASE TENSORIAL DEEP LAKE
# ======================================================================
# Este script permite consultar una base vectorial (tensorial) creada previamente con Deep Lake.
# Realiza búsquedas semánticas basadas en similitud del coseno.
#
# FUNCIONALIDADES:
#   - Carga la configuración desde un archivo JSON.
#   - Abre la base Deep Lake existente (previamente creada).
#   - Permite escribir consultas por consola.
#   - Devuelve los fragmentos más parecidos según embeddings.
#
# REQUISITOS:
#   - Haber ejecutado previamente un script que genere la base tensorial.
#   - Tener configurado el archivo "config.json" con los parámetros adecuados.
# ======================================================================

import os
import json
import numpy as np
from typing import Dict, List, Tuple
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import DeepLake

# ----------------------------------------------------------------------
# CONFIGURACIÓN AMBIENTAL
# ----------------------------------------------------------------------
# Estas variables de entorno desactivan el uso de TensorFlow.
# Evita conflictos y asegura que HuggingFace utilice PyTorch.
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"


# ----------------------------------------------------------------------
# FUNCIÓN: CARGAR CONFIGURACIÓN
# ----------------------------------------------------------------------
def load_config(path: str) -> Dict:
    """
    Carga el archivo JSON de configuración.

    Argumentos:
        path (str): Ruta al archivo config.json

    Retorna:
        Dict: Diccionario con los parámetros definidos en el JSON.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------
# FUNCIÓN: SIMILITUD DEL COSENO
# ----------------------------------------------------------------------
def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calcula la similitud del coseno entre dos vectores.

    Fórmula:
        cos(θ) = (a · b) / (||a|| * ||b||)

    Retorna:
        float: valor entre -1 y 1 donde:
               1 -> vectores muy similares
               0 -> sin relación
              -1 -> opuestos
    """
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12  # Evita división por cero
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ----------------------------------------------------------------------
# FUNCIÓN: RANKING EXPLÍCITO POR COSENO (FALLBACK)
# ----------------------------------------------------------------------
def _rank_with_explicit_cosine(
    embeddings: HuggingFaceEmbeddings,
    query: str,
    docs_text: List[str]
) -> List[Tuple[int, float]]:
    """
    Calcula la similitud del coseno entre el embedding de la consulta
    y los embeddings de cada fragmento.

    Argumentos:
        embeddings: modelo de embeddings HuggingFace
        query (str): texto de la consulta
        docs_text (List[str]): lista de fragmentos de texto

    Retorna:
        List[Tuple[int, float]]: lista [(índice_fragmento, score_coseno), ...] ordenada por score descendente
    """
    # Generar embedding de la consulta
    q_vec = np.array(embeddings.embed_query(query), dtype=np.float32)

    # Generar embeddings de todos los fragmentos
    d_vecs = np.array(embeddings.embed_documents(docs_text), dtype=np.float32)

    # Calcular similitud coseno con cada documento
    scores = [(i, _cosine_similarity(q_vec, d)) for i, d in enumerate(d_vecs)]

    # Ordenar del más similar al menos similar
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


# ----------------------------------------------------------------------
# FUNCIÓN: CONSULTA INTERACTIVA
# ----------------------------------------------------------------------
def interactive_query(cfg: Dict) -> None:
    """
    Inicia un bucle interactivo que permite realizar consultas sobre una base Deep Lake.

    Pasos:
      1. Carga parámetros desde config.json.
      2. Abre la base Deep Lake existente.
      3. Permite ingresar consultas por consola.
      4. Devuelve los fragmentos más similares al texto consultado.
    """

    # --------------------------------------------------------------
    # LECTURA DE PARÁMETROS DESDE CONFIGURACIÓN
    # --------------------------------------------------------------
    dl_cfg = cfg["deeplake"]
    dataset_path = os.path.expanduser(dl_cfg["dataset_path"])  # ruta a la base Deep Lake
    read_only = bool(dl_cfg.get("read_only", True))  # True = solo lectura

    emb_cfg = cfg["embedding"]
    model_name = emb_cfg["model_name"]
    device = emb_cfg.get("device", "cpu")
    normalize_embeddings = bool(emb_cfg.get("normalize_embeddings", True))

    # Cantidad de resultados a recuperar
    k = int(cfg["retrieval"].get("k", 5))

    # --------------------------------------------------------------
    # CARGA DEL MODELO DE EMBEDDINGS
    # --------------------------------------------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},  # CPU o GPU
        encode_kwargs={"normalize_embeddings": normalize_embeddings}
    )

    # --------------------------------------------------------------
    # CONEXIÓN CON LA BASE DEEP LAKE EXISTENTE
    # --------------------------------------------------------------
    print(f"Cargando base Deep Lake desde: {dataset_path} (read_only={read_only})")
    db = DeepLake(dataset_path=dataset_path, embedding=embeddings, read_only=read_only)

    print("\nBase lista para consultas. Escriba 'salir' para terminar.\n")

    # --------------------------------------------------------------
    # BUCLE INTERACTIVO DE CONSULTAS
    # --------------------------------------------------------------
    while True:
        query = input("Consulta: ").strip()

        # Comando de salida
        if query.lower() in {"salir", "exit", "quit"}:
            print("Fin de la sesión de consulta.")
            break

        # Si la consulta está vacía, pedir de nuevo
        if not query:
            continue

        # ----------------------------------------------------------
        # INTENTO PRINCIPAL: búsqueda con puntajes nativos del vector store
        # ----------------------------------------------------------
        try:
            results = db.similarity_search_with_relevance_scores(query, k=k)
            print(f"\nTop {len(results)} resultados:")

            for idx, (doc, score) in enumerate(results, start=1):
                meta = doc.metadata or {}
                page = meta.get("page", "")  # número de página (si existe en metadatos)
                print(f"[{idx}] Score={score:.4f} | Página={page}")
                print(doc.page_content.strip()[:500])  # muestra los primeros 500 caracteres
                print("-" * 80)

        # ----------------------------------------------------------
        # ALTERNATIVA: cálculo explícito de similitud del coseno
        # ----------------------------------------------------------
        except Exception:
            # Recuperar candidatos básicos
            candidates = db.similarity_search(query, k=k)
            texts = [d.page_content for d in candidates]

            # Calcular ranking manualmente por similitud del coseno
            ranking = _rank_with_explicit_cosine(embeddings, query, texts)

            print(f"\nTop {len(ranking)} resultados (coseno explícito):")
            for idx, (i_doc, cos) in enumerate(ranking, start=1):
                doc = candidates[i_doc]
                meta = doc.metadata or {}
                page = meta.get("page", "")
                print(f"[{idx}] Cos={cos:.4f} | Página={page}")
                print(doc.page_content.strip()[:500])
                print("-" * 80)


# ----------------------------------------------------------------------
# BLOQUE PRINCIPAL DE EJECUCIÓN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Ruta del archivo de configuración (por defecto "config.json")
    config_file = "config.json"

    # Carga el archivo de configuración
    cfg = load_config(config_file)

    # Inicia el modo de consulta interactiva
    interactive_query(cfg)
