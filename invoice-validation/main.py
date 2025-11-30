import pdfplumber
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA

# 1. PDF Loader
def load_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Load PDF
pdf_text = load_pdf("invoice.pdf")
print(pdf_text)
documents = [pdf_text]

# 2. Embeddings + Vector DB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.from_texts(documents, embedding=embeddings)
retriever = vector_db.as_retriever(k=5)

# 3. Llama 3 LLM
llm = LlamaCpp(
    model_path="Meta-Llama-3-8B.Q4_K_M.gguf",
    n_ctx=4096,
    max_tokens=2048,
    temperature=0.2,
    n_threads=8
)

# 4. RAG Chain
rag = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)

# 5. Query
query = "Summarize invoice number and total amount."
response = rag(query)

print("ANSWER:\n", response["result"])
print("\nSOURCES:")
for doc in response["source_documents"]:
    print("-", doc.page_content[:200], "...")
