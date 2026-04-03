import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import torch

df = pd.read_csv("turkish_law_dataset.csv")
texts = [f"Soru: {row['soru']} Cevap: {row['cevap']}" for _, row in df.iterrows()]
metadatas = [{"kaynak": row['kaynak'], "cevap": row['cevap']} for _, row in df.iterrows()]
ids = [str(i) for i in range(len(df))]

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('BAAI/bge-m3', device=device)

print(f"Encoding {len(texts)} items on {device}...")
embeddings = model.encode(texts, show_progress_bar=True, batch_size=16)

client = chromadb.PersistentClient(path="./legal_db")
collection = client.get_or_create_collection(name="turkish_law")

print("Saving to database in batches...")
max_batch_size = 5000

for i in range(0, len(ids), max_batch_size):
    batch_ids = ids[i : i + max_batch_size]
    batch_embeddings = embeddings[i : i + max_batch_size].tolist()
    batch_documents = texts[i : i + max_batch_size]
    batch_metadatas = metadatas[i : i + max_batch_size]
    
    collection.add(
        ids=batch_ids,
        embeddings=batch_embeddings,
        documents=batch_documents,
        metadatas=batch_metadatas
    )
    print(f"Uploaded batch {i // max_batch_size + 1}")

print("Done! Your database is ready.")