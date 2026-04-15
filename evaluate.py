import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import ollama
import numpy as np
from sklearn.metrics import precision_recall_curve
import torch
import json
from tqdm import tqdm
from datetime import datetime
import re

EMBED_MODEL_NAME = "BAAI/bge-m3"
OLLAMA_MODEL_NAME = "hf.co/ogulcanaydogan/Turkish-LLM-7B-Instruct-GGUF:Q4_K_M"

device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=device)
client = chromadb.PersistentClient(path="./legal_db")

class RAGEvaluator:
    def __init__(self, test_csv_path="test.csv"):
        self.test_df = pd.read_csv(test_csv_path)
        self.results = []
        
    def retrieve_context(self, query, n_results=3):
        try:
            collection = client.get_collection(name="turkish_law")
        except Exception as e:
            print(f"Error: Database not found. Run embed.py first.")
            return [], []
        
        query_vec = embed_model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=n_results
        )
        
        docs = results['documents'][0] if results['documents'] else []
        metadata = results['metadatas'][0] if results['metadatas'] else []
        
        return docs, metadata
    
    def compute_retrieval_metrics(self, query, expected_context, retrieved_docs, retrieved_meta): # Recall@K, MRR, nDCG
        metrics = {}
    
        relevant_count = 0
        mrr = 0
        for rank, doc in enumerate(retrieved_docs, 1):
            if expected_context.lower() in doc.lower() or self._text_overlap(doc, expected_context) > 0.3:
                relevant_count += 1
                if mrr == 0:
                    mrr = 1 / rank
        
        metrics['recall@3'] = (relevant_count / min(3, len(retrieved_docs))) if retrieved_docs else 0
        metrics['mrr'] = mrr
        
        return metrics
    
    def _text_overlap(self, text1, text2, n_gram=2):
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        overlap = len(words1 & words2)
        return overlap / max(len(words1), len(words2)) if max(len(words1), len(words2)) > 0 else 0
    
    def compute_qa_metrics(self, predicted_answer, expected_answer): # EM, F1, BLEU, ROUGE-L
        # Normalize answers
        pred_tokens = re.findall(r'\b\w+\b', predicted_answer.lower())
        expected_tokens = re.findall(r'\b\w+\b', expected_answer.lower())
        
        metrics = {}

        # EM
        metrics['em'] = 1.0 if self._normalize_answer(predicted_answer) == self._normalize_answer(expected_answer) else 0.0
        
        # F1 Score
        common_tokens = set(pred_tokens) & set(expected_tokens)
        if len(pred_tokens) == 0 and len(expected_tokens) == 0:
            metrics['f1'] = 1.0
        elif len(pred_tokens) == 0 or len(expected_tokens) == 0:
            metrics['f1'] = 0.0
        else:
            precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
            recall = len(common_tokens) / len(expected_tokens) if expected_tokens else 0
            metrics['f1'] = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # BLEU (1-gram + 2-gram precision)
        pred_1grams = set(pred_tokens)
        expected_1grams = set(expected_tokens)
        bleu_1 = len(pred_1grams & expected_1grams) / len(pred_1grams) if pred_1grams else 0
        
        pred_2grams = set(zip(pred_tokens[:-1], pred_tokens[1:]))
        expected_2grams = set(zip(expected_tokens[:-1], expected_tokens[1:]))
        bleu_2 = len(pred_2grams & expected_2grams) / len(pred_2grams) if pred_2grams else 0
        
        metrics['bleu'] = (bleu_1 + bleu_2) / 2
        
        # ROUGE-L
        metrics['rouge_l'] = self._rouge_l(pred_tokens, expected_tokens)
        
        return metrics
    
    def _normalize_answer(self, text):
        return ' '.join(re.findall(r'\b\w+\b', text.lower()))
    
    def _rouge_l(self, pred_tokens, expected_tokens):
        lcs_len = self._lcs_length(pred_tokens, expected_tokens)
        if len(pred_tokens) == 0 or len(expected_tokens) == 0:
            return 0.0
        recall = lcs_len / len(expected_tokens)
        precision = lcs_len / len(pred_tokens)
        return (2 * recall * precision) / (recall + precision) if (recall + precision) > 0 else 0
    
    def _lcs_length(self, list1, list2):
        m, n = len(list1), len(list2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if list1[i-1] == list2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]
    
    def check_citation_accuracy(self, generated_answer, expected_sources):
        expected_sources_list = [s.strip() for s in expected_sources.split(';')] if isinstance(expected_sources, str) else []
        
        cited_sources = []
        for source in expected_sources_list:
            if source.lower() in generated_answer.lower():
                cited_sources.append(source)
        
        accuracy = len(cited_sources) / len(expected_sources_list) if expected_sources_list else 0
        
        return {
            'citation_found': len(cited_sources) > 0,
            'cited_sources': cited_sources,
            'citation_accuracy': accuracy,
            'expected_sources': expected_sources_list
        }
    
    def detect_hallucination(self, generated_answer, retrieved_context):
        if not retrieved_context:
            return 1.0  # High hallucination if no context
        
        answer_tokens = set(re.findall(r'\b\w+\b', generated_answer.lower()))
        context_tokens = set(re.findall(r'\b\w+\b', ' '.join(retrieved_context).lower()))
        
        if len(answer_tokens) == 0:
            return 0.0
        
        overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
        hallucination_score = 1 - overlap
        
        return hallucination_score
    
    def evaluate_baseline(self, sample_size=None, debug=False):
        test_data = self.test_df.head(sample_size) if sample_size else self.test_df
        
        print(f"Evaluating baseline on {len(test_data)} samples...")
        
        for idx, row in tqdm(test_data.iterrows(), total=len(test_data)):
            question = row['soru']
            expected_answer = row['cevap']
            expected_sources = row['kaynak']
            expected_context = row['context']
            
            retrieved_docs, retrieved_meta = self.retrieve_context(question, n_results=3)
            
            retrieval_metrics = self.compute_retrieval_metrics(
                question, expected_context, retrieved_docs, retrieved_meta
            )
            
            try:
                context_text = "\n\n".join(
                    [f"[KAYNAK: {meta['kaynak']}] {doc}" for doc, meta in zip(retrieved_docs, retrieved_meta)]
                )
                
                prompt = f"""Aşağıda verilen hukuk metinlerini temel alarak soruyu cevapla.

            Kurallar:
            1) Sadece verilen hukuk metinlerini kullan. Asla bilgini ekleme.
            2) Cevapta mutlaka kaynak atfı yap. En az bir Kaynak yazmalısın.
            3) Eğer cevap metinlerde yoksa "Bu bilgi verilen kaynaklarda bulunmuyor." yaz.
            4) Uydurma bilgi verme. Sadece verilen metinleri kullan.
            5) Cevaptan sonra, HER ZAMAN "Kaynak:" kısmını yaz.
            6) Formatı AYNI örnekteki gibi tut:

            YANIT FORMATI (zorunlu):
            Yanıt: [Senin cevabın burada olacak]
            Kaynak:
            - [Kanun Adı]

            HUKUKİ METİNLER:
            {context_text}

            SORU: {question}

            CEVAP:
            Yanıt:"""
                
                response = ollama.generate(
                    model=OLLAMA_MODEL_NAME,
                    prompt=prompt,
                    stream=False,
                    options={"temperature": 0.3, "top_p": 0.9}
                )
                generated_answer = response['response'].strip()
                
                has_kaynak = "kaynak:" in generated_answer.lower()
                if not has_kaynak and retrieved_meta:
                    sources = [m.get('kaynak', '') for m in retrieved_meta if m.get('kaynak')]
                    if sources:
                        sources_block = "\n".join(f"- {s}" for s in sources)
                        generated_answer = f"{generated_answer}\n\nKaynak:\n{sources_block}"
                
            except Exception as e:
                print(f"Error generating answer for Q{idx}: {e}")
                generated_answer = "Hata: Cevap üretilemeyen."
            
            # Debug output for first 5 samples
            if debug and idx < 5:
                print(f"\n{'='*80}")
                print(f"[SAMPLE {idx}] Question: {question[:100]}")
                print(f"[SAMPLE {idx}] Expected Sources: {expected_sources}")
                print(f"[SAMPLE {idx}] Retrieved Sources: {[m.get('kaynak', '') for m in retrieved_meta]}")
                print(f"[SAMPLE {idx}] Generated Answer (first 300 chars):\n{generated_answer[:300]}")
                print(f"[SAMPLE {idx}] Has 'Kaynak:' section: {'kaynak:' in generated_answer.lower()}")
                print(f"{'='*80}\n")
            
            # Compute QA metrics
            qa_metrics = self.compute_qa_metrics(generated_answer, expected_answer)
            
            # Check citation accuracy
            citation_metrics = self.check_citation_accuracy(generated_answer, expected_sources)
            
            # Detect hallucination
            hallucination_score = self.detect_hallucination(generated_answer, retrieved_docs)
            
            result = {
                'question_id': idx,
                'question': question,
                'expected_answer': expected_answer,
                'generated_answer': generated_answer,
                'expected_sources': expected_sources,
                'retrieved_sources': [m.get('kaynak', '') for m in retrieved_meta],
                **retrieval_metrics,
                **qa_metrics,
                **citation_metrics,
                'hallucination_score': hallucination_score
            }
            
            self.results.append(result)
        
        return self.results
    
    def aggregate_metrics(self):
        if not self.results:
            print("No results to aggregate. Run evaluate_baseline() first.")
            return {}
        
        df_results = pd.DataFrame(self.results)
        
        aggregated = {
            'total_samples': len(self.results),
            'retrieval_metrics': {
                'avg_recall@3': df_results['recall@3'].mean(),
                'avg_mrr': df_results['mrr'].mean(),
            },
            'qa_metrics': {
                'avg_em': df_results['em'].mean(),
                'avg_f1': df_results['f1'].mean(),
                'avg_bleu': df_results['bleu'].mean(),
                'avg_rouge_l': df_results['rouge_l'].mean(),
            },
            'citation_metrics': {
                'citation_found_rate': df_results['citation_found'].mean(),
                'avg_citation_accuracy': df_results['citation_accuracy'].mean(),
            },
            'hallucination_metrics': {
                'avg_hallucination_score': df_results['hallucination_score'].mean(),
                'hallucination_rate': (df_results['hallucination_score'] > 0.5).mean(),
            }
        }
        
        return aggregated
    
    def save_results(self, output_dir="results"): # Save results to JSON and CSV
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        df_results = pd.DataFrame(self.results)
        csv_path = f"{output_dir}/eval_results_{timestamp}.csv"
        df_results.to_csv(csv_path, index=False)
        
        aggregated = self.aggregate_metrics()
        json_path = f"{output_dir}/eval_metrics_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(aggregated, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Detailed results saved to: {csv_path}")
        print(f"✓ Aggregated metrics saved to: {json_path}")
        
        return aggregated
    
    def print_summary(self):
        aggregated = self.aggregate_metrics()
        
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Total samples evaluated: {aggregated['total_samples']}")
        
        print("\n[RETRIEVAL METRICS]")
        print(f"  Recall@3:  {aggregated['retrieval_metrics']['avg_recall@3']:.4f}")
        print(f"  MRR:       {aggregated['retrieval_metrics']['avg_mrr']:.4f}")
        
        print("\n[QA METRICS]")
        print(f"  EM (Exact Match):  {aggregated['qa_metrics']['avg_em']:.4f}")
        print(f"  F1 Score:          {aggregated['qa_metrics']['avg_f1']:.4f}")
        print(f"  BLEU:              {aggregated['qa_metrics']['avg_bleu']:.4f}")
        print(f"  ROUGE-L:           {aggregated['qa_metrics']['avg_rouge_l']:.4f}")
        
        print("\n[CITATION METRICS]")
        print(f"  Citation Found Rate:      {aggregated['citation_metrics']['citation_found_rate']:.4f}")
        print(f"  Citation Accuracy:        {aggregated['citation_metrics']['avg_citation_accuracy']:.4f}")
        
        print("\n[HALLUCINATION METRICS]")
        print(f"  Avg Hallucination Score:  {aggregated['hallucination_metrics']['avg_hallucination_score']:.4f}")
        print(f"  Hallucination Rate (>0.5):{aggregated['hallucination_metrics']['hallucination_rate']:.4f}")
        print("="*60 + "\n")


if __name__ == "__main__":
    evaluator = RAGEvaluator(test_csv_path="test.csv")
    
    # Quick test with 10 samples:
    #print("Running evaluation on 10 samples with debugging enabled...")
    #evaluator.evaluate_baseline(sample_size=10, debug=True)
    #evaluator.print_summary()
    
    # Full evaluation:
    evaluator.evaluate_baseline()
    metrics = evaluator.save_results()
    evaluator.print_summary()