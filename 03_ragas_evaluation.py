import os
import sys
import json
import warnings
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from config import get_config

warnings.filterwarnings("ignore")

# 1. Environment / imports
load_dotenv()
config = get_config()

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qa_pairs import QA_PAIRS

# 2. Prompt templates (same as step 2)
SYSTEM_V1 = (
    "You are a helpful AI assistant. "
    "Answer the user's question using ONLY the provided context. "
    "Keep your answer concise (2-4 sentences). "
    "If the context does not contain the answer, say: 'I don't have enough information.'\n\n"
    "Context:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([("system", SYSTEM_V1), ("human", "{question}")])

SYSTEM_V2 = (
    "You are an expert AI tutor. Provide a structured, accurate answer.\n\n"
    "Instructions:\n"
    "1. Read the context carefully.\n"
    "2. Identify the key facts relevant to the question.\n"
    "3. Write a clear, well-organized answer (3-5 sentences).\n"
    "4. State explicitly if the context lacks sufficient information.\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([("system", SYSTEM_V2), ("human", "{question}")])

PROMPTS = {
    "v1": PROMPT_V1,
    "v2": PROMPT_V2,
}

# 3. Build vectorstore
def build_vectorstore(embeddings):
    text = Path("data/knowledge_base.txt").read_text()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    return FAISS.from_texts(chunks, embeddings)

# 4. Run RAG and capture outputs + contexts
def run_rag(retriever, llm, prompt, question: str) -> dict:
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    ctx_str = "\n\n".join(contexts)
    
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": ctx_str, "question": question})
    
    return {"answer": answer, "contexts": contexts}

def collect_rag_outputs(vectorstore, prompt_version: str, llm) -> list:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    prompt = PROMPTS[prompt_version]
    
    results = []
    print(f"\nRunning 50 questions with prompt {prompt_version} ...")
    
    for i, qa in enumerate(QA_PAIRS, 1):
        try:
            out = run_rag(retriever, llm, prompt, qa["question"])
            results.append({
                "question":  qa["question"],
                "reference": qa["reference"],
                "answer":    out["answer"],
                "contexts":  out["contexts"],
            })
            print(f"  [{i:02d}/50] {qa['question'][:60]}")
        except Exception as e:
            print(f"  [{i:02d}/50] ❌ Error: {e}")
            
    return results

# 5. Build RAGAS EvaluationDataset
def build_ragas_dataset(rag_results: list):
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]
    return EvaluationDataset(samples=samples)

# 6. Run RAGAS evaluation
def run_ragas_eval(rag_results: list, version: str, llm_eval, emb_eval) -> dict:
    print(f"\n📐 Running RAGAS evaluation for prompt {version} ...")
    
    dataset = build_ragas_dataset(rag_results)
    
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm_eval,
        embeddings=emb_eval,
    )
    
    scores = {}
    for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        raw = result[key]
        scores[key] = float(np.mean([v for v in raw if v is not None]))
    
    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")
        
    return scores

# 7. Main
def main():
    print("=" * 60)
    print("  Step 3: RAGAS Evaluation")
    print("=" * 60)

    # Initialize LLM and Embeddings
    emb = OpenAIEmbeddings(
        model=config["EMBEDDING_MODEL_NAME"],
        openai_api_key=config["OPENAI_API_KEY"],
        base_url=config["OPENAI_BASE_URL"],
    )
    llm = ChatOpenAI(
        model=config["OPENAI_MODEL_NAME"],
        openai_api_key=config["OPENAI_API_KEY"],
        base_url=config["OPENAI_BASE_URL"],
    )

    try:
        vectorstore = build_vectorstore(emb)
        
        # Collect outputs
        v1_results = collect_rag_outputs(vectorstore, "v1", llm)
        v2_results = collect_rag_outputs(vectorstore, "v2", llm)
        
        # Run evaluation
        v1_scores = run_ragas_eval(v1_results, "v1", llm, emb)
        v2_scores = run_ragas_eval(v2_results, "v2", llm, emb)
        
        # Print comparison table
        print("\n" + "-"*60)
        print(f"{'Metric':30s} | {'V1':8s} | {'V2':8s} | Winner")
        print("-"*60)
        for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
            s1, s2 = v1_scores[metric], v2_scores[metric]
            winner = "V1" if s1 > s2 else "V2"
            print(f"{metric:30s} | {s1:.4f} | {s2:.4f} | {winner}")
        print("-"*60)
        
        # Check faithfulness target
        best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
        if best_faith >= 0.8:
            print(f"✅ Target met: faithfulness = {best_faith:.4f}")
        else:
            print(f"⚠️  Below target ({best_faith:.4f}). Try adjusting chunking or prompts.")
            
        # Save JSON report
        report = {
            "prompt_v1_scores": v1_scores,
            "prompt_v2_scores": v2_scores,
            "target_met": best_faith >= 0.8,
        }
        Path("data/ragas_report.json").write_text(json.dumps(report, indent=2))
        print("\n💾 Saved data/ragas_report.json")
        
    except Exception as e:
        print(f"💥 Fatal Error: {e}")

if __name__ == "__main__":
    main()
