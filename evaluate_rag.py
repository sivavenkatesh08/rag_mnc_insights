import json
import csv
from pathlib import Path
from typing import List, Dict
from difflib import SequenceMatcher
from rag_pipeline_gemini import rag_query

# Config
INDEX_PATH = Path(r"C:\22ad053\Navigate Labs\rag_mnc_insights\data\outputs\mnc_faiss_index")
EVAL_FILE = Path(r"C:\22ad053\Navigate Labs\rag_mnc_insights\evaluation_samples.json")


def keyword_match(answer: str, keywords: List[str]) -> float:
    """Returns the proportion of expected keywords found in the answer."""
    matches = sum(1 for kw in keywords if kw.lower() in answer.lower())
    return matches / len(keywords) if keywords else 0.0


def similarity_score(a: str, b: str) -> float:
    """Computes a rough similarity score between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def evaluate_rag(samples: List[Dict]):
    results = []
    for idx, sample in enumerate(samples):
        print(f"\n🔎 Evaluating Sample {idx+1}: {sample['question']}")
        answer = rag_query(INDEX_PATH, sample['question'], fiscal_filter=True, return_answer_only=True)

        kw_score = keyword_match(answer, sample.get("expected_keywords", []))
        sim_score = similarity_score(answer, sample.get("expected_answer", ""))

        results.append({
            "question": sample['question'],
            "keywords_matched": kw_score,
            "text_similarity": sim_score,
            "rag_answer": answer
        })

    return results


def save_results(results: List[Dict], filename="evaluation_results.csv"):
    keys = results[0].keys()
    with open(filename, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✅ Results saved to: {filename}")


if __name__ == "__main__":
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        samples = json.load(f)

    results = evaluate_rag(samples)
    save_results(results)
