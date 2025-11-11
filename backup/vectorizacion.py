import PyPDF2
from sentence_transformers import SentenceTransformer, util
import os
import torch
import argparse
import time
import glob
import requests

# --- CONFIGURACIÓN ---
PDF_DIR = "files"
#SEARCH_QUERY = "QA Automation , performance , gherkin,CI/CD , Python, Fullstack, Selenium, BDD ,API Testing, Mobile Testing, Appium, Jenkins, Agile, Scrum, TDD, DevOps, Test Automation Engineer , SDET, Automation Architect, Test Lead, Test Manager, Quality Assurance Engineer, Manual Testing, Software Testing, Software Quality Assurance, Test Analyst, Test Engineer, Test Consultant, QA Engineer, QA Analyst, QA Consultant, QA Lead, QA Manager, Quality Engineer, Quality Analyst, Quality Consultant, Quality Lead, Quality Manager"
SEARCH_QUERY = "Experiencia en Python y librerias que incluyan Python"
API_URL = "http://127.0.0.1:8000/compare/"

# --- FUNCIONES MODULARIZADAS ---

def extract_text_from_pdf(pdf_path: str) -> str | None:
    """Extrae todo el texto de un archivo PDF."""
    if not os.path.exists(pdf_path):
        print(f"Error: El archivo '{pdf_path}' no fue encontrado.")
        return None
    print(f"🔍 Extrayendo texto del PDF: {os.path.basename(pdf_path)}...")
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

# ... (Las funciones de embeddings no cambian) ...
def generate_openai_embeddings(texts: list[str], model_name: str) -> list[list[float]]:
    from openai import OpenAI
    from dotenv import load_dotenv
    print(f"📐 Vectorizando texto usando la API de OpenAI con el modelo '{model_name}'...")
    try:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key: raise ValueError("OPENAI_API_KEY no encontrada.")
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(input=texts, model=model_name)
        embeddings = [item.embedding for item in response.data]
        print("✅ Vectorización con OpenAI completada.")
        return embeddings
    except Exception as e:
        print(f"💥 Error al generar embeddings con OpenAI: {e}")
        return None

def generate_gemini_embeddings(texts: list[str], model_name: str) -> list[list[float]]:
    import google.generativeai as genai
    from dotenv import load_dotenv
    print(f"📐 Vectorizando texto usando la API de Gemini con el modelo '{model_name}'...")
    try:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key: raise ValueError("GEMINI_API_KEY no encontrada.")
        genai.configure(api_key=api_key)
        response = genai.embed_content(model=f"models/{model_name}", content=texts, task_type="retrieval_document")
        print("✅ Vectorización con Gemini completada.")
        return response['embedding']
    except Exception as e:
        print(f"💥 Error al generar embeddings con Gemini: {e}")
        return None

def generate_sentence_transformer_embeddings(texts: str | list[str], model: SentenceTransformer) -> torch.Tensor:
    print(f"📐 Vectorizando texto usando el modelo LOCAL...")
    return model.encode(texts, convert_to_tensor=True)

def evaluate_similarity_score(score: float) -> tuple[str, str]:
    if score >= 0.80: return ("🟢 Coincidencia Excelente", "El perfil es una combinación casi perfecta. ¡Contactar de inmediato!")
    elif score >= 0.60: return ("🟢 Coincidencia Alta", "El perfil es muy relevante. Debe ser considerado una alta prioridad.")
    elif score >= 0.40: return ("🟡 Coincidencia Moderada", "El perfil es prometedor y está relacionado. Vale la pena revisarlo.")
    elif score >= 0.20: return ("🟡 Coincidencia Baja", "El perfil tiene alguna relación, pero es probable que no sea el ideal.")
    else: return ("🔴 Muy Baja Coincidencia", "El perfil no es relevante para la búsqueda.")

# --- NUEVA FUNCIÓN UNIFICADA DE ANÁLISIS ---
def analyze_pdf(pdf_path: str, search_query: str, query_embedding: torch.Tensor, model=None, embedding_method: str = "sentence", use_chunking: bool = False) -> tuple[float, list[str]]:
    """
    Analiza un PDF usando el método y estrategia de chunking especificados.
    Devuelve el mejor score y las palabras clave encontradas.
    """
    profile_text = extract_text_from_pdf(pdf_path)
    if not profile_text:
        return None, []

    score_value = None

    # --- LÓGICA CUANDO CHUNKING ESTÁ ACTIVADO ---
    if use_chunking:
        print("   🧩 Estrategia de Chunking Activada.")
        
        if embedding_method == 'sentence':
            # Chunking eficiente con modelo local
            chunks = [chunk.strip() for chunk in profile_text.split('.') if len(chunk.strip()) > 15]
            if not chunks: return None, []
            print(f"   🧩 Analizando {len(chunks)} fragmentos del CV...")
            chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
            cosine_scores = util.cos_sim(query_embedding, chunk_embeddings)
            score_value = cosine_scores.max().item()

        elif embedding_method in ['openai', 'gemini']:
            # Chunking ineficiente y costoso con APIs
            print("   ⚠️  ADVERTENCIA: Usar chunking con APIs de terceros puede ser lento y costoso.")
            chunks = [chunk.strip() for chunk in profile_text.split('.') if len(chunk.strip()) > 15]
            if not chunks: return None, []
            
            best_score = -1
            api_func = generate_openai_embeddings if embedding_method == 'openai' else generate_gemini_embeddings
            model_name = "text-embedding-3-small" if embedding_method == 'openai' else "text-embedding-004"
            
            for i, chunk in enumerate(chunks):
                print(f"   📡 Llamando a la API para el fragmento {i+1}/{len(chunks)}...")
                chunk_embedding_list = api_func([chunk], model_name)
                if chunk_embedding_list:
                    chunk_embedding = torch.tensor(chunk_embedding_list[0])
                    cosine_score = util.cos_sim(query_embedding, chunk_embedding)
                    if cosine_score.item() > best_score:
                        best_score = cosine_score.item()
            score_value = best_score

        elif embedding_method == 'api':
            # Chunking no soportado por la API personalizada
            print("   ℹ️  AVISO: El servidor API no soporta chunking. Se usará el análisis de documento completo.")
            # Se cae en la lógica de abajo (sin chunking)

    # --- LÓGICA CUANDO CHUNKING ESTÁ DESACTIVADO (O NO SOPORTADO) ---
    if score_value is None:
        print("   📄 Estrategia de Documento Completo.")
        if embedding_method == 'sentence':
            profile_embedding = generate_sentence_transformer_embeddings(profile_text, model)
        elif embedding_method == 'openai':
            embeddings_list = generate_openai_embeddings([profile_text], "text-embedding-3-small")
            if not embeddings_list: return None, []
            profile_embedding = torch.tensor(embeddings_list[0])
        elif embedding_method == 'gemini':
            embeddings_list = generate_gemini_embeddings([profile_text], "text-embedding-004")
            if not embeddings_list: return None, []
            profile_embedding = torch.tensor(embeddings_list[0])
        elif embedding_method == 'api':
            print("   📡 Enviando datos al servidor API...")
            try:
                payload = {"cv_text": profile_text, "search_query": search_query}
                response = requests.post(API_URL, json=payload)
                response.raise_for_status()
                api_result = response.json()
                score_value = api_result.get("similarity_score", 0.0)
                print("   ✅ Respuesta recibida del servidor.")
            except requests.exceptions.RequestException as e:
                print(f"   💥 Error al conectar con la API: {e}")
                return None, []
        
        if embedding_method != 'api':
            cosine_score = util.cos_sim(profile_embedding, query_embedding)
            score_value = cosine_score.item()

    # Búsqueda de palabras clave (común para todos)
    keywords = [kw.strip().lower() for kw in search_query.replace(',', ' ').replace('y ', ' ').split()]
    profile_text_lower = profile_text.lower()
    found_keywords = [kw for kw in keywords if kw in profile_text_lower]
    
    return score_value, found_keywords

# --- FUNCIÓN PRINCIPAL ---
def main():
    start_time_total = time.time()

    parser = argparse.ArgumentParser(description="Evalúa la compatibilidad de CVs usando diferentes motores de IA o una API.")
    parser.add_argument('--embeddings', type=str, required=True, choices=['sentence', 'openai', 'gemini', 'api'], help="El motor de embeddings a usar.")
    parser.add_argument('--chunked', type=str, required=True, choices=['true', 'false'], help="Usar análisis por fragmentos (chunking) o por documento completo.")
    
    args = parser.parse_args()
    embedding_method = args.embeddings
    use_chunking = args.chunked.lower() == 'true'

    print(f"🚀 Iniciando análisis con el motor: '{embedding_method}'")
    print(f"🔍 Chunking: {'Activado' if use_chunking else 'Desactivado'}")
    print(f"📁 Analizando PDFs en el directorio: '{PDF_DIR}'")
    print(f"🔍 Búsqueda: '{SEARCH_QUERY}'")

    if not os.path.exists(PDF_DIR):
        print(f"Error: El directorio '{PDF_DIR}' no fue encontrado.")
        return

    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    if not pdf_files:
        print(f"No se encontraron archivos PDF en el directorio '{PDF_DIR}'.")
        return

    print(f"Se encontraron {len(pdf_files)} archivos PDF para procesar.")

    load_time = 0.0
    embed_time = 0.0
    query_embedding = None
    model = None

    if embedding_method != 'api':
        if embedding_method == 'sentence':
            model_name = 'hiiamsid/sentence_similarity_spanish_es'
            start_time_load = time.time()
            print(f"\n🧠 Cargando modelo de IA: {model_name}...")
            model = SentenceTransformer(model_name)
            end_time_load = time.time()
            load_time = end_time_load - start_time_load
            print(f"✅ Modelo cargado en {load_time:.2f} segundos.")

        start_time_embed = time.time()
        if embedding_method == 'sentence':
            query_embedding = generate_sentence_transformer_embeddings(SEARCH_QUERY, model)
        elif embedding_method == 'openai':
            embeddings_list = generate_openai_embeddings([SEARCH_QUERY], "text-embedding-3-small")
            if not embeddings_list: return
            query_embedding = torch.tensor(embeddings_list[0])
        elif embedding_method == 'gemini':
            embeddings_list = generate_gemini_embeddings([SEARCH_QUERY], "text-embedding-004")
            if not embeddings_list: return
            query_embedding = torch.tensor(embeddings_list[0])
        
        end_time_embed = time.time()
        embed_time = end_time_embed - start_time_embed
        print(f"✅ Vector de consulta generado en {embed_time:.2f} segundos.")

    results = []
    processing_times = []
    
    for pdf_path in pdf_files:
        start_time_process = time.time()
        print(f"\n🔬 Procesando: {os.path.basename(pdf_path)}")
        
        # Llamada a la nueva función unificada
        score_value, found_keywords = analyze_pdf(pdf_path, SEARCH_QUERY, query_embedding, model, embedding_method, use_chunking)

        if score_value is not None:
            category, recommendation = evaluate_similarity_score(score_value)
            results.append({
                'filename': os.path.basename(pdf_path),
                'score': score_value,
                'category': category,
                'recommendation': recommendation,
                'found_keywords': found_keywords
            })
            
            end_time_process = time.time()
            processing_times.append(end_time_process - start_time_process)

    results.sort(key=lambda x: x['score'], reverse=True)

    # ... (El resto del código para mostrar resultados y análisis de rendimiento no cambia) ...
    print("\n" + "="*60)
    print("                  RESULTADOS DE LA COMPARACIÓN")
    print("="*60)
    
    for i, result in enumerate(results, 1):
        print(f"\n📄 Perfil #{i}: {result['filename']}")
        print("-" * 60)
        print(f"📊 Similitud del Coseno (Score): {result['score']:.4f}")
        print(f"📈 Porcentaje de Relevancia: {result['score']:.2%}")
        print(f"Categoría: {result['category']}")
        print(f"Recomendación: {result['recommendation']}")
        print(f"✅ Palabras Clave Encontradas: {', '.join(result['found_keywords']) if result['found_keywords'] else 'Ninguna'}")
    
    print("="*60)

    end_time_total = time.time()
    total_time = end_time_total - start_time_total
    avg_process_time = sum(processing_times) / len(processing_times) if processing_times else 0

    print("\n" + "="*60)
    print("                     ANÁLISIS DE RENDIMIENTO")
    print("="*60)
    if embedding_method == 'api':
        print(f"⏱️ Método de procesamiento: API Remota")
    elif embedding_method == 'sentence':
        print(f"⏱️ Tiempo de carga del modelo (LOCAL): {load_time:.4f} segundos")
        print(f"⏱️ Tiempo de generación de vector de consulta (LOCAL): {embed_time:.4f} segundos")
    else:
        print(f"⏱️ Tiempo de generación de vector de consulta (API): {embed_time:.4f} segundos")
    
    print(f"⏱️ Tiempo promedio de procesamiento por CV: {avg_process_time:.4f} segundos")
    print("-" * 60)
    print(f"🏁 TIEMPO TOTAL DE EJECUCIÓN: {total_time:.4f} segundos")
    print(f"📊 CVs procesados: {len(results)} de {len(pdf_files)}")
    print("="*60)

if __name__ == "__main__":
    main()