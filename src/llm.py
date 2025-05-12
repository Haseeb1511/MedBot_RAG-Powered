from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import EnsembleRetriever,ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereRerank
from langchain_groq import ChatGroq
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from dotenv import load_dotenv
from vector_store import text_splitter,doc_load
load_dotenv()


path_to_data = "data/"
document = doc_load(path_to_data)
chunks = text_splitter(document)

parser = StrOutputParser()
prompt_text = PromptTemplate(
    template="""You are a highly accurate medical assistant.
        Use ONLY the given context to answer the user's question.
        If the context does not contain the information needed, simply reply:
        "I don't know based on the given context."
        CONTEXT:
        {context}
        QUESTION:
        {question}
        Your Answer:""",
input_variables=["context", "question"])


callback = [StreamingStdOutCallbackHandler()]
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
model = ChatGroq(model="llama-3.1-8b-instant",streaming=True,callbacks=callback)
model_image = ChatGoogleGenerativeAI(model = "models/gemini-pro-vision")


vector_store = PineconeVectorStore(
    index_name="multimodal-rag",
    embedding=embedding_model,
)


bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k=3

dense_retriever = vector_store.as_retriever(search_kwargs={"k":3})

hybrid_retrieval = EnsembleRetriever(
    retrievers=[bm25_retriever,dense_retriever],
    weights=[0.5,0.5]
)

reranker = CohereRerank(model = "rerank-english-v3.0")

final_retriever = ContextualCompressionRetriever(
    base_retriever=hybrid_retrieval,
    base_compressor=reranker
)



def context_format(context):
    return "\n\n".join(doc.page_content for doc in context)

parallel_chain = RunnableParallel({
      "context": final_retriever | RunnableLambda(context_format),
      "question":RunnablePassthrough()
  })
text_chain = parallel_chain | prompt_text | model | parser