import streamlit as st
import pickle
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks.base import BaseCallbackHandler


# Load API Key
os.environ["GOOGLE_API_KEY"] = "Your API key"

# ===========================================================================================
# Sidebar contents
with st.sidebar:
    st.title(" 📎Chat With any PDF or Text File")
    st.markdown('''
    Assessment Round - 10 Minute School AI Engineer (Level 1) 📜.
    ''')
    add_vertical_space(5)
    st.write('Made with by RASEL [AI Engineer](https://github.com/RaselSarker606)')

# ===========================================================================================
# Main App
def main():
    st.header('📎 Chat with PDF or Text Files')

    # Upload PDF
    pdf = st.file_uploader("Upload your PDF or Text File", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Text splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # Embeddings + FAISS
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            st.success(" Embedding Loaded from Cache")
        else:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
            st.success(" Embedding Created and Saved")
# ===========================================================================================
        # User query
        query = st.text_input("🔍 Ask a question from the PDF:")

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            doc_text = "\n\n".join([doc.page_content for doc in docs])

            # LLM Setup
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",
                temperature=0.1
            )

            # Chain with Custom Prompt
            chain = LLMChain(llm=llm, prompt=custom_prompt)
            response = chain.run({"docs": doc_text, "query": query})
            st.write("📌Response:")
            st.info(response)

# ===========================================================================================

# Custom PromptTemplate
custom_prompt = PromptTemplate(
    template="""
🌐 You are an intelligent multilingual assistant capable of understanding both বাংলা and English.

🧠 Your task is to answer the user's query strictly based on the provided docs below.

⚠️ Guidelines:
- Use only the given docs to answer.
- If the answer is not found in the docs, reply with "জানি না" or "I don't know" depending on the question language.
- Keep your answer very short and direct (1 sentence maximum).
- Do not generate or assume any information outside the context.
- If the query is in English, respond in English. If the query is in Bengali, respond in Bengali.

📚 Context:
{docs}

❓ Question:
{query}
""",
    input_variables=["docs", "query"]
)

# ===========================================================================================
if __name__ == '__main__':
    main()
