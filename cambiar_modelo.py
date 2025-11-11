import requests
import argparse

# --- CONFIGURACIÓN DE LA API ---
API_URL = "http://127.0.0.1:8000/change_model/"

def main():
    """
    Función principal que solicita al servidor API cambiar el modelo de IA.
    """
    # 1. Configurar el parser de argumentos
    parser = argparse.ArgumentParser(
        description="Solicita a la API del servidor que cambie su modelo de IA activo."
    )
    
    # 2. Definir el argumento --model
    parser.add_argument(
        '--model', 
        type=str, 
        required=True, 
        help="El nombre del modelo a cargar en el servidor (ej: 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2')."
    )
    
    # 3. Parsear los argumentos de la línea de comandos
    args = parser.parse_args()
    nuevo_modelo = args.model

    print(f"🔄 Solicitando al servidor que cambie al modelo: '{nuevo_modelo}'")

    try:
        # 4. Realizar la petición POST al endpoint de cambio de modelo
        response = requests.post(API_URL, params={"model_name": nuevo_modelo})
        
        # Lanza un error si la petición falló (código 4xx o 5xx)
        response.raise_for_status()
        
        # 5. Procesar y mostrar la respuesta del servidor
        result = response.json()
        print(f"✅ Respuesta del servidor: {result.get('message')}")
        
    except requests.exceptions.RequestException as e:
        print(f"💥 Error al conectar con la API: {e}")
        print("   Asegúrate de que el servidor (api_server.py) está corriendo y accesible.")

if __name__ == "__main__":
    main()