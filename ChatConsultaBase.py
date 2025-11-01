import os
import json
import time
import numpy as np
import warnings
from typing import Dict, List, Tuple
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import DeepLake
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.memory import ConversationSummaryMemory

# ------------------------------------------------------------
# CONFIGURACIÓN INICIAL
# ------------------------------------------------------------
warnings.filterwarnings("ignore")
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ------------------------------------------------------------
# FUNCIÓN: CARGAR CONFIGURACIÓN
# ------------------------------------------------------------
def load_config(path: str) -> Dict:
    """Carga y valida el archivo de configuración JSON."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


# ------------------------------------------------------------
# FUNCIÓN: CALCULAR SIMILITUD DEL COSENO
# ------------------------------------------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Devuelve la similitud del coseno entre dos vectores."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom) if denom != 0 else 0.0


# ------------------------------------------------------------
# FUNCIÓN: BUSCAR EL EMBEDDING MÁS SIMILAR
# ------------------------------------------------------------
def retrieve_top_embedding(cfg: Dict, query: str) -> Tuple[str, float]:
    """
    Busca el fragmento más similar a la consulta en la base Deep Lake.
    Devuelve el texto del fragmento y su score de similitud.
    """

    # Parámetros de la configuración
    dl_cfg = cfg["deeplake"]
    emb_cfg = cfg["embedding"]
    k = int(cfg["retrieval"].get("k", 5))

    # Cargar embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=emb_cfg["model_name"],
        model_kwargs={"device": emb_cfg.get("device", "cpu")},
        encode_kwargs={"normalize_embeddings": bool(emb_cfg.get("normalize_embeddings", True))}
    )

    dataset_path = os.path.expanduser(dl_cfg["dataset_path"])
    db = DeepLake(dataset_path=dataset_path, embedding=embeddings, read_only=dl_cfg.get("read_only", True))

    # Buscar los k más similares
    results = db.similarity_search(query, k=k)
    docs_text = [r.page_content for r in results]

    # Calcular similitud explícita
    q_vec = np.array(embeddings.embed_query(query), dtype=np.float32)
    d_vecs = np.array(embeddings.embed_documents(docs_text), dtype=np.float32)

    scores = [cosine_similarity(q_vec, d) for d in d_vecs]
    best_idx = int(np.argmax(scores))
    best_text = docs_text[best_idx]
    best_score = scores[best_idx]

    print(f"\nTop {k} resultados por similitud del coseno:")
    for i, s in enumerate(scores):
        print(f"[{i+1}] cos={s:.4f}")
    print("-" * 80)

    return best_text, best_score


# ------------------------------------------------------------
# FUNCIÓN PRINCIPAL: CHAT CON RAG Y MEMORIA
# ------------------------------------------------------------
def chat_with_rag(cfg: Dict):
    """Inicia un chat que usa Deep Lake como base de contexto y memoria conversacional resumida."""

    # Cargar API key
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No se encontró OPENAI_API_KEY en el archivo .env")

    # Configurar LLM
    llm_cfg = cfg["llm"]
    llm = ChatOpenAI(
        openai_api_base=llm_cfg["api_base"],
        openai_api_key=api_key,
        model_name=llm_cfg["model_name"],
        temperature=float(llm_cfg["temperature"])
    )

    # Configurar memoria conversacional
    mem_cfg = cfg.get("memory", {})
    memory = None
    if mem_cfg.get("enabled", False):
        print(f"Memoria conversacional activada ({mem_cfg['type']}) con modelo: {mem_cfg['summary_model']}")
        memory = ConversationSummaryMemory(
            llm=llm,
            return_messages=True,
            max_token_limit=int(mem_cfg.get("max_summary_length", 1000))
        )

    system_instruction = cfg.get("system_instruction", "Responde de forma clara y precisa.")

    print("\nRAG con memoria conversacional listo. Escribe 'salir' para terminar.\n")

    while True:
        user_input = input("👤 Usuario: ").strip()
        if user_input.lower() in {"salir", "exit", "quit"}:
            print("Fin de la sesión.")
            break
        if not user_input:
            continue

        # Recuperar el mejor fragmento desde la base
        best_text, score = retrieve_top_embedding(cfg, user_input)

        # Construir prompt combinado
        Meta_prompt = (
            f"{system_instruction}\n\n"
            f"-----------------------------------------------------------------"
            f"Contexto más relevante (similitud={score:.4f}):\n{best_text}\n\n"
            f"-----------------------------------------------------------------"
            f"Pregunta del usuario:\n{user_input}"
        )

        # Agregar memoria previa si existe
        messages = []
        if memory:
            memory.save_context({"input": user_input}, {"output": ""})
            messages = memory.chat_memory.messages

        # Enviar al modelo
        try:
            response = llm.invoke([HumanMessage(content=Meta_prompt)])
            print("\n🤖 Respuesta del asistente:\n")
            print(response.content.strip())
            print("-" * 80)

            # Actualizar memoria con la respuesta
            if memory:
                memory.save_context({"input": user_input}, {"output": response.content.strip()})

            time.sleep(1)

        except Exception as e:
            print(f"Error al consultar el modelo: {e}")


# ------------------------------------------------------------
# PUNTO DE ENTRADA
# ------------------------------------------------------------
if __name__ == "__main__":
    config_file = "config.json"
    cfg = load_config(config_file)
    chat_with_rag(cfg)
