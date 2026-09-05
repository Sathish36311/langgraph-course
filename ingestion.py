# Load Articles to documents
# Chunk documents
# Embed
# Store in ChromaDB vector store

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_unstructured import UnstructuredLoader
from langchain_openai import OpenAIEmbeddings

load_dotenv()


urls = [ "https://lilianweng.github.io/posts/2023-06-23-agent/",
         "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
         "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [UnstructuredLoader(web_url=url, chunking_strategy="basic", max_character=1000000).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_overlap=0, chunk_size=250)
doc_splits = text_splitter.split_documents(docs_list)

# vector_store = Chroma.from_documents(
#     documents=doc_splits,
#     embedding=OpenAIEmbeddings(),
#     collection_name='rag-chroma',
#     persist_directory='./.chroma'
# )

retriever = Chroma(
    embedding_function=OpenAIEmbeddings(),
    collection_name='rag-chroma',
    persist_directory='./.chroma'
).as_retriever()
