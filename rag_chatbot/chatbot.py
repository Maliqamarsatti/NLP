# rag_chatbot/chatbot.py
import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# 1) Load and split the knowledge base
def init_chatbot():
    HERE = os.path.dirname(__file__)
    KB_PATH = os.path.join(HERE, "coffee_diseases.txt")  # ensure this file exists in rag_chatbot/
    loader = TextLoader(KB_PATH)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    # 2) Build FAISS retriever (k=5)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(split_docs, embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    # 3) Load the LLM pipeline (Google FLAN-T5-Small)
    model_id = "google/flan-t5-small"
    gen_pipe = pipeline(
        "text2text-generation",
        model=model_id,
        tokenizer=model_id,
        max_length=512,
        do_sample=False,
        device_map="auto",
    )
    llm = HuggingFacePipeline(pipeline=gen_pipe)

    # 4) Custom prompt for detailed multi-sentence answers
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are an expert plant pathologist. Use only the provided CONTEXT to answer.\n"
            "If the context does NOT contain the information needed, reply exactly:\n\n"
            "    \"I’m sorry, I don’t have enough information to answer that.\"\n\n"
            "Otherwise, answer in a detailed, step-by-step manner with practical tips.\n"
            "Be sure to explain why each step is recommended and provide at least three sentences.\n\n"
            "CONTEXT:\n{context}\n\n"
            "QUESTION:\n{question}\n\n"
            "ANSWER:"
        )
    )

    # 5) Build RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain


def ask_question(query: str, qa_chain) -> str:
    """Run a retrieval-augmented query requesting detailed explanation."""
    wrapped = (
        "Please provide a detailed, multi-sentence explanation with reasoning: " + query
    )
    return qa_chain.run(wrapped).strip()

