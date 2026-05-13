import os
from pathlib import Path
from dotenv import load_dotenv
from config import get_config

# 1. Environment setup
load_dotenv()
config = get_config()

# Set LangSmith environment variables
os.environ["LANGCHAIN_TRACING_V2"] = config["LANGCHAIN_TRACING_V2"]
os.environ["LANGCHAIN_API_KEY"] = config["LANGCHAIN_API_KEY"] or ""
os.environ["LANGCHAIN_PROJECT"] = config["LANGCHAIN_PROJECT"]
os.environ["LANGCHAIN_ENDPOINT"] = config["LANGCHAIN_ENDPOINT"]

# 2. Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable
from qa_pairs import QA_PAIRS

# 3. LLM and Embeddings
llm = ChatOpenAI(
    model=config["OPENAI_MODEL_NAME"],
    openai_api_key=config["OPENAI_API_KEY"],
    base_url=config["OPENAI_BASE_URL"],
)

embeddings = OpenAIEmbeddings(
    model=config["EMBEDDING_MODEL_NAME"],
    openai_api_key=config["OPENAI_API_KEY"],
    base_url=config["OPENAI_BASE_URL"],
)

# 4. Build FAISS vector store
def build_vectorstore():
    kb_path = Path("data/knowledge_base.txt")
    if not kb_path.exists():
        raise FileNotFoundError("Knowledge base file not found at data/knowledge_base.txt")
    
    text = kb_path.read_text()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    print(f"Split into {len(chunks)} chunks")
    
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

# 5. RAG prompt template
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the context below to answer the user question. If the context doesn't contain the answer, say you don't know.\n\nContext:\n{context}"),
    ("human", "{question}"),
])

# 6. Build the RAG chain
def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever

# 7. Traced query function
@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    return chain.invoke(question)

# 9. Main
def main():
    print("=" * 60)
    print("  Step 1: LangSmith RAG Pipeline")
    print("=" * 60)

    try:
        vectorstore = build_vectorstore()
        chain, retriever = build_rag_chain(vectorstore)
        
        questions = [qa["question"] for qa in QA_PAIRS]
        
        for i, question in enumerate(questions, 1):
            try:
                answer = ask(chain, question)
                print(f"[{i:02d}/{len(questions)}] Q: {question[:60]}")
                print(f"       A: {answer[:100]}\n")
            except Exception as e:
                print(f"[{i:02d}/{len(questions)}] ❌ Error: {e}")

        print(f"✅ {len(questions)} traces sent to LangSmith project '{os.environ['LANGCHAIN_PROJECT']}'")
        print("   Open https://smith.langchain.com to view traces.")
        
    except Exception as e:
        print(f"💥 Fatal Error: {e}")

if __name__ == "__main__":
    main()
