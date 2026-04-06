import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import torch

df = pd.read_csv("turkish_law_dataset.csv")

required_columns = {"context", "kaynak"}
missing_columns = required_columns - set(df.columns)
if missing_columns:
    raise ValueError(f"CSV is missing required columns: {missing_columns}")

df = df.dropna(subset=["context", "kaynak"]).copy()
df["context"] = df["context"].astype(str).str.strip()
df["kaynak"] = df["kaynak"].astype(str).str.strip()
df = df[(df["context"] != "") & (df["kaynak"] != "")]

if "soru" in df.columns:
    grouped = (
        df.groupby(["kaynak", "context"], as_index=False)
        .agg(
            qa_count=("soru", "count"),
        )
    )
else:
    grouped = (
        df.groupby(["kaynak", "context"], as_index=False)
        .size()
        .rename(columns={"size": "qa_count"})
    )

qa_examples_by_group = {}
if "soru" in df.columns and "cevap" in df.columns:
    qa_df = df.dropna(subset=["soru", "cevap"]).copy()
    qa_df["soru"] = qa_df["soru"].astype(str).str.strip()
    qa_df["cevap"] = qa_df["cevap"].astype(str).str.strip()
    qa_df = qa_df[(qa_df["soru"] != "") & (qa_df["cevap"] != "")]

    for (kaynak, context), group in qa_df.groupby(["kaynak", "context"]):
        examples = []
        for _, row in group.head(3).iterrows():
            examples.append(f"Soru: {row['soru']} | Cevap: {row['cevap']}")
        qa_examples_by_group[(kaynak, context)] = "\n".join(examples)

texts = []
metadatas = []
ids = []

for i, row in grouped.iterrows():
    kaynak = row["kaynak"]
    context = row["context"]
    texts.append(f"KAYNAK: {kaynak}\n\nHUKUK METNI:\n{context}")
    metadatas.append(
        {
            "kaynak": kaynak,
            "qa_count": int(row["qa_count"]),
            "qa_examples": qa_examples_by_group.get((kaynak, context), ""),
        }
    )
    ids.append(str(i))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('BAAI/bge-m3', device=device)

print(f"Encoding {len(texts)} items on {device}...")
embeddings = model.encode(texts, show_progress_bar=True, batch_size=16)

client = chromadb.PersistentClient(path="./legal_db")
try:
    client.delete_collection(name="turkish_law")
except Exception:
    pass
collection = client.get_or_create_collection(name="turkish_law")

print("Saving to database in batches...")
max_batch_size = 5000

for i in range(0, len(ids), max_batch_size):
    batch_ids = ids[i : i + max_batch_size]
    batch_embeddings = embeddings[i : i + max_batch_size].tolist()
    batch_documents = texts[i : i + max_batch_size]
    batch_metadatas = metadatas[i : i + max_batch_size]
    
    collection.upsert(
        ids=batch_ids,
        embeddings=batch_embeddings,
        documents=batch_documents,
        metadatas=batch_metadatas
    )
    print(f"Uploaded batch {i // max_batch_size + 1}")

print("Done! Your database is ready.")