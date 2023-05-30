from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain import HuggingFaceHub
from langchain.chains import RetrievalQAWithSourcesChain
from blog_loader import get_blog_documents, CHINESE, ENGLISH_SEPARATORS

import os


def load_from_chinese_translation_blogs():
    """Load from the Chinese translation of the book."""
    docs = []
    url = "https://www.qinyu.info/post/wardley-maps/ch{i}/"
    [docs.extend(get_blog_documents(url.format(i=i+1),
                                    CHINESE)) for i in range(6)]
    return docs


def load_from_original_book(book_path):
    """Load from the original book."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=200,
                                              separators=ENGLISH_SEPARATORS)
    docs = UnstructuredPDFLoader(book_path).load_and_split(
        text_splitter=splitter)

    print("Loaded {num} documents".format(num=len(docs)))
    print("Example documents:")
    for doc in docs[:5]:
        print("{start}..{end}".format(start=doc.page_content[:50], end=doc.page_content[-50:]))
        print(len(doc.page_content))
        # print(doc.metadata)
    return docs


persist_directory = 'db'
embedding = HuggingFaceEmbeddings()


def load_db(persist_directory, embedding):
    """Load the vector database."""
    if os.path.exists(persist_directory):
        return Chroma(persist_directory=persist_directory,
                      embedding_function=embedding)
    else:
        docs = load_from_original_book("books/wardley_maps.pdf")
        vectordb = Chroma.from_documents(documents=docs,
                                         embedding=embedding,
                                         persist_directory=persist_directory)
        vectordb.persist()
        return vectordb


vectordb = load_db(persist_directory, embedding)
# try similarity search
print(vectordb.similarity_search_with_score("What is a wardley map?"))

# just works
llm = HuggingFaceHub(repo_id="gpt2-xl",
                     model_kwargs={'temperature': 0.5, 'max_length': 150})

qa = RetrievalQAWithSourcesChain.from_chain_type(
     llm=llm,
     chain_type="stuff",
     retriever=vectordb.as_retriever())

qa({"question": "What is a map?"}, return_only_outputs=True)
