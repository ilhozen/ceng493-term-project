import chromadb
from sentence_transformers import SentenceTransformer
import ollama
import sys

print("Loading bge-m3 retrieval model...")
embed_model = SentenceTransformer('BAAI/bge-m3')

client = chromadb.PersistentClient(path="./legal_db")

try:
    collection = client.get_collection(name="turkish_law")
except Exception as e:
    print("Error: Database 'turkish_law' not found. Run embed.py first.")
    sys.exit()

def ask_legal_question(user_query):
    print(f"\nSearching for: {user_query}")
    query_vec = embed_model.encode(user_query).tolist()
    
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=3
    )
    
    context_list = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        context_list.append(f"[KAYNAK: {meta['kaynak']}] {doc}")
    
    context = "\n\n".join(context_list)
    
    prompt = f"""Aşağıda verilen hukuk metinlerini temel alarak soruyu cevapla. 
Cevabını oluştururken mutlaka ilgili kaynağa (Örn: TCK Madde X) atıf yap. 
Eğer metinde cevap yoksa, bilmediğini belirt ve uydurma.

HUKUKİ METİNLER:
{context}

SORU: {user_query}
CEVAP:"""

    print("Generating answer with turkish-llm-7b...")
    response = ollama.generate(
        model='hf.co/ogulcanaydogan/Turkish-LLM-7B-Instruct-GGUF:Q4_K_M', 
        prompt=prompt
    )
    
    return response['response']

if __name__ == "__main__":
    test_question = "Hırsızlık suçunun cezası nedir?"
    
    final_answer = ask_legal_question(test_question)
    
    print("-" * 30)
    print("SİSTEM CEVABI:")
    print(final_answer)
    print("-" * 30)