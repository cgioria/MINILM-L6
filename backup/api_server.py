# api_server.py
from fastapi import FastAPI
from pydantic import BaseModel
import PyPDF2
import torch
from sentence_transformers import SentenceTransformer, util
import time

# --- 1. CARGAR EL MODELO UNA SOLA VEZ AL INICIAR EL SERVIDOR ---
print("🚀 Iniciando servidor de IA...")
print("🧠 Cargando modelo de IA (esto puede tardar un momento)...")
model = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')
print("✅ Modelo cargado y listo para usar.")

# --- 2. CREAR LA APLICACIÓN FASTAPI ---
app = FastAPI(
    title="API de Comparación de CVs",
    description="Una API para evaluar la compatibilidad entre un CV y una búsqueda usando IA."
)

# --- 3. DEFINIR EL FORMATO DE LOS DATOS DE ENTRADA ---
class ComparisonRequest(BaseModel):
    cv_text: str
    search_query: str

# --- 4. CREAR EL ENDPOINT (LA URL) PARA LA COMPARACIÓN ---
@app.post("/compare/")
def compare_profiles(request: ComparisonRequest):
    """
    Recibe el texto de un CV y una búsqueda, y devuelve un score de compatibilidad.
    """
    start_time = time.time()

    # Usar el modelo ya cargado para generar los vectores
    profile_embedding = model.encode(request.cv_text, convert_to_tensor=True)
    query_embedding = model.encode(request.search_query, convert_to_tensor=True)
    
    # Calcular similitud
    cosine_score = util.cos_sim(profile_embedding, query_embedding)
    score_value = cosine_score.item()
    
    end_time = time.time()
    total_time = end_time - start_time

    # Devolver el resultado en formato JSON
    return {
        "model_used": "hiiamsid/sentence_similarity_spanish_es",
        "similarity_score": score_value,
        "relevance_percentage": f"{score_value:.2%}",
        "processing_time_seconds": f"{total_time:.4f}"
    }

# --- 5. (Opcional) Endpoint de salud para verificar que el servidor está vivo ---
@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": True}
