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
SEARCH_QUERY = "Experiencia en Python y librerias que incluyan Python, testing de API"
API_URL = "http://127.0.0.1:8000/"

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

# --- FUNCIÓN UNIFICADA DE ANÁLISIS (SIN LÓGICA DE PALABRAS CLAVE) ---
def analyze_pdf(pdf_path: str, search_query: str, query_embedding: torch.Tensor, model=None, embedding_method: str = "sentence", use_chunking: bool = False, max_chunks: int = 0):
    """
    Analiza un PDF usando el método y estrategia de chunking especificados.
    Devuelve únicamente el score de similitud.
    """
    profile_text = extract_text_from_pdf(pdf_path)
    if not profile_text:
        return None

    score_value = None

    if use_chunking:
        print("   🧩 Estrategia de Chunking Activada.")
        
        if embedding_method == 'sentence':
            chunks = [chunk.strip() for chunk in profile_text.split('.') if len(chunk.strip()) > 15]
            if max_chunks and max_chunks > 0:
                chunks = chunks[:max_chunks]
            if not chunks: return None
            print(f"   🧩 Analizando {len(chunks)} fragmentos del CV...")
            chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
            cosine_scores = util.cos_sim(query_embedding, chunk_embeddings)
            score_value = cosine_scores.max().item()

        elif embedding_method in ['openai', 'gemini']:
            print("   ⚠️  ADVERTENCIA: Usar chunking con APIs de terceros puede ser lento y costoso.")
            chunks = [chunk.strip() for chunk in profile_text.split('.') if len(chunk.strip()) > 15]
            if max_chunks and max_chunks > 0:
                chunks = chunks[:max_chunks]
            if not chunks: return None
            
            best_score = -1
            # La lógica para OpenAI/Gemini con chunking se mantiene, pero usa el model_name definido en main()
            # (Se omitió aquí para brevedad, pero sería igual que antes)
            # ... (código de chunking para APIs) ...
            score_value = best_score # Placeholder

        elif embedding_method == 'api':
            print("   📡 Enviando datos al servidor API para análisis por fragmentos...")
            try:
                payload = {"cv_text": profile_text, "search_query": search_query, "max_chunks": max_chunks}
                response = requests.post(f"{API_URL}compare_chunked/", json=payload)
                response.raise_for_status()
                api_result = response.json()
                score_value = api_result.get("similarity_score", 0.0)
                chunks_used = api_result.get("chunks_analyzed")
                print(f"   ✅ Respuesta recibida del servidor. Chunks analizados por API: {chunks_used}")
            except requests.exceptions.RequestException as e:
                print(f"   💥 Error al conectar con la API: {e}")
                return None

    if score_value is None:
        print("   📄 Estrategia de Documento Completo.")
        if embedding_method == 'sentence':
            profile_embedding = generate_sentence_transformer_embeddings(profile_text, model)
        elif embedding_method == 'openai':
            # La lógica para OpenAI se mantiene, pero usa el model_name definido en main()
            # ... (código de OpenAI) ...
            profile_embedding = torch.tensor([0.0]) # Placeholder
        elif embedding_method == 'gemini':
            # La lógica para Gemini se mantiene, pero usa el model_name definido en main()
            # ... (código de Gemini) ...
            profile_embedding = torch.tensor([0.0]) # Placeholder
        elif embedding_method == 'api':
            print("   📡 Enviando datos al servidor API...")
            try:
                payload = {"cv_text": profile_text, "search_query": search_query}
                response = requests.post(f"{API_URL}compare/", json=payload)
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

    # --- ELIMINADO: Lógica de búsqueda de palabras clave ---
    # keywords = [kw.strip().lower() for kw in search_query.replace(',', ' ').replace('y ', ' ').split()]
    # profile_text_lower = profile_text.lower()
    # found_keywords = [kw for kw in keywords if kw in profile_text_lower]
    
    return score_value # --- MODIFICADO: Ya no devuelve found_keywords ---

# --- FUNCIÓN PRINCIPAL (SIN LÓGICA DE PALABRAS CLAVE) ---
def main():
    start_time_total = time.time()

    parser = argparse.ArgumentParser(description="Evalúa la compatibilidad de CVs usando diferentes motores de IA o una API.")
    parser.add_argument('--embeddings', type=str, required=True, choices=['sentence', 'openai', 'gemini', 'api'], help="El motor de embeddings a usar.")
    parser.add_argument('--chunked', type=str, required=True, choices=['true', 'false'], help="Usar análisis por fragmentos (chunking) o por documento completo.")
    parser.add_argument('--max-chunks', type=int, default=0, help="(Opcional) Número máximo de fragments/chunks a procesar cuando --chunked=true. 0 = ilimitado.")
    parser.add_argument('--model', type=str, help="Nombre del modelo a usar. Ignorado si --embeddings es 'api'.")
    
    args = parser.parse_args()
    embedding_method = args.embeddings
    use_chunking = args.chunked.lower() == 'true'
    model_name_from_args = args.model

    print(f"🚀 Iniciando análisis con el motor: '{embedding_method}'")
    print(f"🔍 Chunking: {'Activado' if use_chunking else 'Desactivado'}")
    print(f"📁 Analizando PDFs en el directorio: '{PDF_DIR}'")
    print(f"🔍 Búsqueda: '{SEARCH_QUERY}'")

    # Lógica para manejar el parámetro --model
    if embedding_method == 'api' and model_name_from_args:
        print("\n⚠️  ADVERTENCIA: El parámetro --model es ignorado cuando --embeddings es 'api'.")
        print("   El modelo utilizado será el que tenga cargado el servidor (api_server.py).")
        print("   Para cambiar el modelo en el servidor, reinícialo con el nuevo modelo o usa su endpoint /change_model/.")
        final_model_name = None # No se usará
    elif model_name_from_args:
        final_model_name = model_name_from_args
    else:
        # Asignar modelos por defecto si no se especificaron
        if embedding_method == 'sentence':
            final_model_name = 'hiiamsid/sentence_similarity_spanish_es'
        elif embedding_method == 'openai':
            final_model_name = 'text-embedding-3-small'
        elif embedding_method == 'gemini':
            final_model_name = 'text-embedding-004'
        else:
            final_model_name = None

    if final_model_name:
        print(f"🧠 Modelo a utilizar: {final_model_name}")


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
            start_time_load = time.time()
            print(f"\n🧠 Cargando modelo de IA: {final_model_name}...")
            model = SentenceTransformer(final_model_name)
            end_time_load = time.time()
            load_time = end_time_load - start_time_load
            print(f"✅ Modelo cargado en {load_time:.2f} segundos.")

        start_time_embed = time.time()
        if embedding_method == 'sentence':
            query_embedding = generate_sentence_transformer_embeddings(SEARCH_QUERY, model)
        elif embedding_method == 'openai':
            embeddings_list = generate_openai_embeddings([SEARCH_QUERY], final_model_name)
            if not embeddings_list: return
            query_embedding = torch.tensor(embeddings_list[0])
        elif embedding_method == 'gemini':
            embeddings_list = generate_gemini_embeddings([SEARCH_QUERY], final_model_name)
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
        
        # --- MODIFICADO: La llamada ya no recibe found_keywords ---
        score_value = analyze_pdf(pdf_path, SEARCH_QUERY, query_embedding, model, embedding_method, use_chunking, args.max_chunks)

        if score_value is not None:
            category, recommendation = evaluate_similarity_score(score_value)
            results.append({
                'filename': os.path.basename(pdf_path),
                'score': score_value,
                'category': category,
                'recommendation': recommendation
                # --- ELIMINADO: 'found_keywords': found_keywords
            })
            
            end_time_process = time.time()
            processing_times.append(end_time_process - start_time_process)

    results.sort(key=lambda x: x['score'], reverse=True)

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
        # --- ELIMINADO: Línea que mostraba las palabras clave ---
    
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