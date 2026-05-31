"""
LLM Generator for FedRAG evaluation pipeline.
Adds the missing LLM step: retrieved docs + query -> generated answer
"""
import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"

def generate_answer(query: str, retrieved_docs: list[str], model: str = "mistral") -> str:
    """
    Takes query + top-K retrieved docs, generates answer via LLM.
    This is the step that was missing in your evaluation.
    """
    context = "\n\n".join(retrieved_docs)
    prompt = f"""You are a factual QA assistant. Answer the question using ONLY the context below.
Give a short, direct answer (1-3 words or a letter like A/B/C/D if it's multiple choice).

Context:
{context}

Question: {query}

Answer:"""

    response = requests.post(OLLAMA_URL, json={
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 20}
    }, timeout=60)
    
    result = response.json()
    return result["response"].strip()


def generate_answers_batch(queries: list[str], retrieved_docs_list: list[list[str]], 
                            model: str = "mistral") -> list[str]:
    """Batch version — runs all queries through the LLM."""
    answers = []
    for i, (query, docs) in enumerate(zip(queries, retrieved_docs_list)):
        print(f"  LLM generating answer {i+1}/{len(queries)}...", end="\r")
        ans = generate_answer(query, docs, model)
        answers.append(ans)
    print()
    return answers
