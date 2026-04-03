# CENG493 Term Project Setup

## 1) Create and activate virtual environment

```powershell
py -m venv rag_env
.\rag_env\Scripts\Activate.ps1
```

If activation is blocked, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\rag_env\Scripts\Activate.ps1
```

## 2) Install dependencies

```powershell
pip install -r requirements.txt
```

## 3) Build embeddings database

Make sure your `turkish_law_dataset.csv` file is in the project root, then run:

```powershell
python embed.py
```

This creates/updates the local Chroma database in `./legal_db`.

## 7) Ask a test query

```powershell
python query.py
```

## Notes

- `query.py` expects the `turkish_law` collection to exist, so run `embed.py` first.
- `query.py` uses Ollama with model:
  - `hf.co/ogulcanaydogan/Turkish-LLM-7B-Instruct-GGUF:Q4_K_M`
- If needed, make sure Ollama is installed and running before querying.

## Deactivate environment

```powershell
deactivate
```
