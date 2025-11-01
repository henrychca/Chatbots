import os
import json
from typing import Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import DeepLake

# Evita que Transformers intente usar TensorFlow si no lo necesitas
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"


def load_config(path: str) -> Dict:
    """Carga el archivo JSON de configuración."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_deeplake_db(cfg: Dict) -> None:
    """Crea (o sobreescribe) la base tensorial Deep Lake a partir de un PDF."""
    pdf_path = os.path.expanduser(cfg["pdf_path"])
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"No existe el PDF indicado: {pdf_path}")

    dl_cfg = cfg["deeplake"]
    dataset_path = os.path.expanduser(dl_cfg["dataset_path"])
    overwrite = bool(dl_cfg.get("overwrite", False))

    emb_cfg = cfg["embedding"]
    model_name = emb_cfg["model_name"]
    device = emb_cfg.get("device", "cpu")
    normalize_embeddings = bool(emb_cfg.get("normalize_embeddings", True))

    split_cfg = cfg["splitter"]
    chunk_size = int(split_cfg["chunk_size"])
    chunk_overlap = int(split_cfg["chunk_overlap"])
    separators = split_cfg.get("separators", ["\n\n", "\n", ".", " ", ""])

    print(f"Cargando PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"Total de páginas cargadas: {len(docs)}")

    print(f"Dividiendo en fragmentos (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )
    split_docs = splitter.split_documents(docs)
    print(f"Total de fragmentos generados: {len(split_docs)}")

    print(f"Cargando modelo de embeddings: {model_name} (device={device})")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": normalize_embeddings}
    )

    print(f"Creando base Deep Lake en: {dataset_path} (overwrite={overwrite})")
    DeepLake.from_documents(
        documents=split_docs,
        embedding=embeddings,
        dataset_path=dataset_path,
        overwrite=overwrite
    )

    print("Base Deep Lake creada correctamente.")


if __name__ == "__main__":
    config_file = "config.json"
    cfg = load_config(config_file)
    build_deeplake_db(cfg)
