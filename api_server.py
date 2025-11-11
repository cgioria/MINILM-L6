from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import torch
from sentence_transformers import SentenceTransformer, util
import time

# --- 1. CARGAR EL MODELO UNA SOLA VEZ AL INICIAR EL SERVIDOR ---
print("🚀 Iniciando servidor de IA...")

# Función para cargar modelo usando la caché de Hugging Face
def load_model(model_name="hiiamsid/sentence_similarity_spanish_es"):
    """
    Carga un modelo usando SentenceTransformer.
    La biblioteca se encarga de descargarlo y cachearlo automáticamente
    en el directorio de caché de Hugging Face (~/.cache/huggingface/hub).
    """
    print(f"🧠 Cargando modelo: {model_name}...")
    print("   (El modelo se descargará automáticamente si no está en la caché)")
    try:
        model = SentenceTransformer(model_name)
        print("✅ Modelo cargado y listo para usar.")
        return model
    except Exception as e:
        print(f"💥 Error al cargar el modelo {model_name}: {e}")
        raise

# Variable global para el modelo y su nombre
model = load_model()
CURRENT_MODEL_NAME = "hiiamsid/sentence_similarity_spanish_es"

# --- 2. CREAR LA APLICACIÓN FASTAPI ---
app = FastAPI(
    title="API de Comparación de CVs",
    description="Una API para evaluar la compatibilidad entre un CV y una búsqueda usando IA."
)

# --- 3. DEFINIR EL FORMATO DE LOS DATOS DE ENTRADA ---
class ComparisonRequest(BaseModel):
    cv_text: str
    search_query: str
    # Opcionales para controlar chunking desde el cliente
    max_chunks: Optional[int] = 0
    chunk_method: Optional[str] = "dot"
    chunk_size: Optional[int] = 500
    top_n: Optional[int] = 1
    model_name: Optional[str] = None

# --- 4. ENDPOINT PARA CAMBIAR EL MODELO ---
@app.post("/change_model/")
def change_model(model_name: str):
    """
    Cambia el modelo utilizado para generar embeddings.
    El modelo se cargará (o se tomará de la caché) y se usará globalmente.
    """
    global model, CURRENT_MODEL_NAME
    try:
        model = load_model(model_name)
        CURRENT_MODEL_NAME = model_name
        return {
            "status": "success", 
            "message": f"Modelo global cambiado a {model_name}"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al cambiar el modelo: {str(e)}")

# --- 5. ENDPOINT DE COMPARACIÓN ---
@app.post("/compare/")
def compare_profiles(request: ComparisonRequest):
    """
    Recibe el texto de un CV y una búsqueda, y devuelve un score de compatibilidad.
    """
    start_time = time.time()
    
    # Determinar qué modelo usar
    current_model = model
    model_to_report = CURRENT_MODEL_NAME

    if request.model_name and request.model_name != CURRENT_MODEL_NAME:
        try:
            print(f"🔄 Cargando modelo específico para esta petición: {request.model_name}")
            current_model = load_model(request.model_name)
            model_to_report = request.model_name
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error al cargar el modelo especificado: {str(e)}")

    # Usar el modelo para generar los vectores
    profile_embedding = current_model.encode(request.cv_text, convert_to_tensor=True)
    query_embedding = current_model.encode(request.search_query, convert_to_tensor=True)
    
    # Calcular similitud
    cosine_score = util.cos_sim(profile_embedding, query_embedding)
    score_value = cosine_score.item()
    
    end_time = time.time()
    total_time = end_time - start_time

    return {
        "model_used": model_to_report,
        "similarity_score": score_value,
        "relevance_percentage": f"{score_value:.2%}",
        "processing_time_seconds": f"{total_time:.4f}"
    }

# --- NUEVO ENDPOINT PARA SOPORTAR CHUNKING ---
@app.post("/compare_chunked/")
def compare_profiles_chunked(request: ComparisonRequest):
    # ... (La lógica de este endpoint es la misma, solo hay que adaptar la parte de carga del modelo) ...
    start_time = time.time()
    
    current_model = model
    model_to_report = CURRENT_MODEL_NAME

    if request.model_name and request.model_name != CURRENT_MODEL_NAME:
        try:
            print(f"🔄 Cargando modelo específico para chunking: {request.model_name}")
            current_model = load_model(request.model_name)
            model_to_report = request.model_name
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error al cargar el modelo especificado: {str(e)}")

    method = (request.chunk_method or "dot").lower()
    size = request.chunk_size or 500
    text = request.cv_text or ""

    query_embedding = current_model.encode(request.search_query, convert_to_tensor=True)

    if method == 'dot':
        chunks = [chunk.strip() for chunk in text.split('.') if len(chunk.strip()) > 15]
    elif method == 'chars':
        chunks = [text[i:i+size].strip() for i in range(0, len(text), size)]
        chunks = [c for c in chunks if len(c) > 15]
    elif method == 'words':
        words = text.split()
        if not words:
            chunks = []
        else:
            chunks = [' '.join(words[i:i+size]).strip() for i in range(0, len(words), size)]
            chunks = [c for c in chunks if len(c) > 15]
    else:
        chunks = [chunk.strip() for chunk in text.split('.') if len(chunk.strip()) > 15]

    if not chunks:
        return {"error": "El texto del CV es demasiado corto o no contiene fragmentos válidos."}

    if request.max_chunks and request.max_chunks > 0:
        chunks = chunks[:request.max_chunks]

    chunk_embeddings = current_model.encode(chunks, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, chunk_embeddings)
    scores_1d = cosine_scores[0] if cosine_scores.dim() == 2 else cosine_scores

    if scores_1d.numel() == 0:
        return {"error": "No se pudieron calcular similitudes; embeddings vacíos."}

    top_n = max(1, int(request.top_n or 1))
    sorted_indices = torch.argsort(scores_1d, descending=True)
    top_indices = sorted_indices[:top_n].tolist()
    top_scores = [float(scores_1d[i].item()) for i in top_indices]
    top_chunks = [chunks[i] for i in top_indices]
    best_score = top_scores[0]

    end_time = time.time()
    total_time = end_time - start_time

    return {
        "model_used": model_to_report,
        "similarity_score": best_score,
        "relevance_percentage": f"{best_score:.2%}",
        "processing_time_seconds": f"{total_time:.4f}",
        "chunks_analyzed": len(chunks),
        "top_n": top_n,
        "top_scores": top_scores,
        "top_chunks": top_chunks
    }

# --- 6. Endpoint de salud ---
@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": True, "current_model": CURRENT_MODEL_NAME}