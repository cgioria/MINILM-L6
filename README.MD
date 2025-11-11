

OPENAI
1)python vectorizacion.py --embeddings openai     --chunked true --max-chunks 5  

SENTENCE
1)python vectorizacion.py --embeddings sentence     --chunked true --max-chunks 6  

GEMINI
1)python vectorizacion.py --embeddings gemini  --chunked true   --max-chunks 10  

SENTENCE (baja y carga del modelo desde hugging face)
1)python vectorizacion.py --embeddings sentence  --model "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  --chunked true   --max-chunks 5  

API model (se carga el modelo descargado localmente como API)
1)uvicorn api_server:app --reload
2)#Cambiar modelo (opcional)
python cambiar_modelo.py --model "hiiamsid/sentence_similarity_spanish_es"
3)python vectorizacion.py --embeddings api  --chunked true --max-chunks 7  






