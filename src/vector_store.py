from pinecone import Pinecone,ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from tqdm import tqdm
load_dotenv()


path_to_data = "data/"
def doc_load(path):
    return DirectoryLoader(path,glob="*.pdf",loader_cls=PyPDFLoader).load()
document = doc_load(path_to_data)
print("len:",len(document))


def text_splitter(doc):
    return RecursiveCharacterTextSplitter(chunk_size=900,chunk_overlap=220).split_documents(doc)
chunks = text_splitter(document)
print("len of chunk:",len(chunks))




if __name__=="__main__":
    def connecting_to_pinecone():
        api = os.getenv("PINECONE_API_KEY")
        env = os.getenv("PINECONE_ENV")

        pc = Pinecone(api_key=api)

        if "multimodal-rag" not in pc.list_indexes().names():
            pc.create_index(
                dimension=768,
                name="multimodal-rag",
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        return "Connected to pinecone succesfully"



    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


    def create_vector_store(batch_size):
        for i in tqdm (range(0,len(chunks),batch_size)):
            batch = chunks[i:i+batch_size]
            vector_store = PineconeVectorStore.from_documents(
                documents=batch,
                embedding=embedding_model,
                index_name="multimodal-rag")
            return "vector store created successfully"
        
    vector_store = create_vector_store(50)
