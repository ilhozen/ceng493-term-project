import chromadb
from sentence_transformers import SentenceTransformer
import ollama
import sys
import torch

EMBED_MODEL_NAME = "BAAI/bge-m3"
OLLAMA_MODEL_NAME = "hf.co/ogulcanaydogan/Turkish-LLM-7B-Instruct-GGUF:Q4_K_M"

print("Loading bge-m3 retrieval model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=device)

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
    source_list = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        context_list.append(f"[KAYNAK: {meta['kaynak']}] {doc}")
        source_list.append(meta['kaynak'])

    unique_sources = list(dict.fromkeys(source_list))

    context = "\n\n".join(context_list)
    
    prompt = f"""Aşağıda verilen hukuk metinlerini temel alarak soruyu cevapla.
Kurallar:
1) Sadece verilen hukuk metinlerini kullan.
2) Cevapta mutlaka kaynak atfı yap.
3) Eğer cevap metinlerde yoksa "Bu bilgi verilen kaynaklarda bulunmuyor." yaz.
4) Uydurma bilgi verme.
5) Soruyu yeniden yazma, paraphrase etme veya yeni bir soru üretme.
6) Cevabı doğrudan ver; "Madde:" gibi başlıklarla yeni soru cümlesi yazma.

YANIT FORMATI (zorunlu):
Yanıt: <kısa ve net açıklama>
Kaynak:
- <kaynak 1>
- <kaynak 2>

HUKUKİ METİNLER:
{context}

SORU: {user_query}
CEVAP:"""

    print("Generating answer with turkish-llm-7b...")
    response = ollama.generate(
        model=OLLAMA_MODEL_NAME,
        prompt=prompt
    )

    response_text = response['response'].strip()
    has_citation = ("kaynak" in response_text.lower()) or any(
        source in response_text for source in unique_sources
    )

    if not has_citation and unique_sources:
        sources_block = "\n".join(f"- {source}" for source in unique_sources)
        response_text += f"\n\nKullanılan Kaynaklar:\n{sources_block}"

    return response_text

if __name__ == "__main__":
    test_question = "Birisi yaşadığı yerin dışında öldüğünde, ölüm yeri ile ilgili yetkililerin sorumlulukları nelerdir?"
    
    final_answer = ask_legal_question(test_question)
    
    print("-" * 30)
    print("SİSTEM CEVABI:")
    print(final_answer)
    print("-" * 30)