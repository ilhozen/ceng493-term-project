import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

df = pd.read_csv("turkish_law_dataset.csv")

model = SentenceTransformer('BAAI/bge-m3')
client = chromadb.PersistentClient(path="./legal_db")
collection = client.get_or_create_collection(name="turkish_law")

for i, row in df.iterrows():
    combined_text = f"Soru: {row['soru']} Cevap: {row['cevap']}"
    embedding = model.encode(combined_text).tolist()
    
    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[combined_text],
        metadatas=[{"kaynak": row['kaynak'], "cevap": row['cevap']}]
    )