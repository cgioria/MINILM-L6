import PyPDF2
import requests
import argparse
import time
import os

# --- CONFIGURACIÓN DE LA API ---
API_URL = "http://127.0.0.1:8000/compare/"
CHANGE_MODEL_URL = "http://127.0.0.1:8000/change_model/"

# --- FUNCIONES ---

def extract_text_from_pdf(pdf_path: str) -> str | None:
    """Extrae todo el texto de un archivo PDF."""
    if not os.path.exists(pdf_path):
        print(f"Error: El archivo '{pdf_path}' no fue encontrado.")
        return None
    print("🔍 Extrayendo texto del PDF...")
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
        print("✅ Texto extraído con éxito.")
        return text
    except Exception as e:
        print(f"💥 Ocurrió un error al leer el PDF: {e}")
        return None

def change_model(model_name: str) -> bool:
    """
    Cambia el modelo utilizado en el servidor API.
    """
    print(f"🔄 Solicitando cambio de modelo a '{model_name}'...")
    try:
        response = requests.post(CHANGE_MODEL_URL, params={"model_name": model_name})
        response.raise_for_status()
        result = response.json()
        print(f"✅ {result.get('message', 'Modelo cambiado correctamente')}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"💥 Error al cambiar el modelo: {e}")
        return False

def call_api(cv_text: str, search_query: str, model_name: str = None) -> dict | None:
    """
    Envía los datos a la API y devuelve la respuesta JSON.
    """
    payload = {
        "cv_text": cv_text,
        "search_query": search_query
    }
    
    if model_name:
        payload["model_name"] = model_name
        print(f"🧠 Usando modelo específico: {model_name}")
    
    print("📡 Enviando datos al servidor API...")
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        print("✅ Respuesta recibida del servidor.")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"💥 Error al conectar con la API: {e}")
        print("Asegúrate de que el servidor (api_server.py) está corriendo en http://127.0.0.1:8000")
        return None

def evaluate_similarity_score(score: float) -> tuple[str, str]:
    """Clasifica el score de similitud en una categoría y una recomendación."""
    if score >= 0.80:
        return ("🟢 Coincidencia Excelente", "El perfil es una combinación casi perfecta. ¡Contactar de inmediato!")
    elif score >= 0.60:
        return ("🟢 Coincidencia Alta", "El perfil es muy relevante. Debe ser considerado una alta prioridad.")
    elif score >= 0.40:
        return ("🟡 Coincidencia Moderada", "El perfil es prometedor y está relacionado. Vale la pena revisarlo.")
    elif score >= 0.20:
        return ("🟡 Coincidencia Baja", "El perfil tiene alguna relación, pero es probable que no sea el ideal.")
    else:
        return ("🔴 Muy Baja Coincidencia", "El perfil no es relevante para la búsqueda.")

# --- FUNCIÓN PRINCIPAL ---
def main():
    """Función principal que orquesta el proceso del cliente."""
    start_time_total = time.time()

    parser = argparse.ArgumentParser(description="Cliente para evaluar la compatibilidad de un CV usando una API local.")
    parser.add_argument('--pdf', type=str, required=True, help="Ruta al archivo PDF del CV.")
    parser.add_argument('--query', type=str, required=True, help="Texto de la búsqueda o requerimiento del puesto.")
    parser.add_argument('--embeddings', type=str, choices=['api'], help="Usar embeddings a través de la API.")
    parser.add_argument('--model', type=str, help="Nombre del modelo a usar (ej: hiiamsid/sentence_similarity_spanish_es).")
    
    args = parser.parse_args()
    pdf_path = args.pdf
    search_query = args.query
    use_embeddings_api = args.embeddings == 'api'
    model_name = args.model

    print(f"🚀 Iniciando cliente de análisis de CVs")
    print(f"📄 Analizando el PDF: '{pdf_path}'")
    print(f"🔍 Búsqueda: '{search_query}'")
    
    if use_embeddings_api:
        print(f"🧠 Usando modo de embeddings a través de la API")
    
    # 1. Extraer texto del PDF
    profile_text = extract_text_from_pdf(pdf_path)
    if not profile_text:
        return

    # 2. Cambiar el modelo si se especificó
    if model_name:
        if not change_model(model_name):
            return

    # 3. Llamar a la API
    api_result = call_api(profile_text, search_query, model_name)
    if not api_result:
        return

    # 4. Procesar y mostrar los resultados
    score_value = api_result.get("similarity_score", 0.0)
    processing_time = api_result.get("processing_time_seconds", "N/A")
    model_used = api_result.get("model_used", "N/A")

    # Análisis de palabras clave (lado del cliente)
    keywords = [kw.strip().lower() for kw in search_query.replace(',', ' ').replace('y ', ' ').split()]
    profile_text_lower = profile_text.lower()
    found_keywords = [kw for kw in keywords if kw in profile_text_lower]
    
    category, recommendation = evaluate_similarity_score(score_value)

    print("\n" + "="*60)
    print("                  RESULTADO DE LA COMPARACIÓN")
    print("="*60)
    print(f"📄 Perfil Analizado: {os.path.basename(pdf_path)}")
    print(f"🔍 Búsqueda Ingresada: '{search_query}'")
    print("-" * 60)
    print(f"📊 Similitud del Coseno (Score): {score_value:.4f}")
    print(f"📈 Porcentaje de Relevancia: {score_value:.2%}")
    print("="*60)
    
    print("\n" + "="*60)
    print("                     EVALUACIÓN DEL RESULTADO")
    print("="*60)
    print(f"Categoría: {category}")
    print(f"Recomendación: {recommendation}")
    print("-" * 60)
    print(f"Palabras Clave de la Búsqueda: {', '.join(keywords)}")
    print(f"✅ Palabras Clave Encontradas en el CV: {', '.join(found_keywords) if found_keywords else 'Ninguna'}")
    print("="*60)

    # --- SECCIÓN FINAL: RESUMEN DE RENDIMIENTO ---
    end_time_total = time.time()
    total_time = end_time_total - start_time_total

    print("\n" + "="*60)
    print("                     ANÁLISIS DE RENDIMIENTO")
    print("="*60)
    print(f"🤖 Modelo usado en el servidor: {model_used}")
    print(f"⏱️ Tiempo de procesamiento en el servidor: {processing_time} segundos")
    print(f"⏱️ Tiempo total del cliente (incl. llamada API): {total_time:.4f} segundos")
    print("="*60)


if __name__ == "__main__":
    main()