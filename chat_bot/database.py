import os
import json
import chromadb
from collections import defaultdict

from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

class ChromaDB():
    def __init__(self, data_name):
        self.client = chromadb.PersistentClient()
        self.collection = self.client.get_or_create_collection(
            name="data",
            metadata={"hnsw:space": "cosine"}
        )
        self.data_dir = "./data"
        self.n_results = 3
        self.initialize(data_name)

    def read_data(self, filename):
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, "r") as f:
            lines = f.readlines()
        return lines

    def initialize(self, data_name):
        print(f"db initialize - {data_name}")
        filename = f"project_data_{data_name}.txt"
        lines = self.read_data(filename)
        manual_dict = self.get_manual_dict(lines)
        self.add_data_to_db(manual_dict)

    def get_manual_dict(self, lines):
        manual_dict = defaultdict(list)
        manual_key = ""
        for line in lines:
            line = line.replace("\n", "")
            if not line:
                continue

            if line[0] == "#":
                manual_key = line.replace("#", "")
            elif manual_key:
                manual_dict[manual_key].append(line)
            else:
                continue

        for key in manual_dict:
            manual_dict[key] = "\n".join(manual_dict[key])

        return manual_dict

    def add_data_to_db(self, manual_data):
        ids = []
        doc_meta = []
        docs = []
        for key, val in manual_data.items():
            ids.append(key)
            docs.append(val)

        self.collection.add(
            documents=docs,
            ids=ids,
        )

    def get_data_from_db(self, query):
        results = self.collection.query(
            query_texts=[query],
            n_results=self.n_results,
        )
        print(f"db query result: {results}")

        ids = results["ids"][0]
        documents = results["documents"][0]

        ans_data = {}
        for i in range(len(ids)):
            ans_data[ids[i]] = documents[i]

        print("answer dict")
        for key, val in ans_data.items():
            print(f"{key}: {val}")

        return json.dumps(ans_data)

class VectorDB():
    def __init__(self, data_dir="./data", upload=True):
        self.data_dir = data_dir
        self.chroma_persist_dir = os.path.join(data_dir, "upload/chroma-persist")
        self.chroma_collection_name = "dosu-bot"
        self.db = None
        self.initialize(upload)

    def initialize(self, upload):
        self.db = Chroma(
            persist_directory=self.chroma_persist_dir,
            embedding_function=OpenAIEmbeddings(),
            collection_name=self.chroma_collection_name,
        )
        if upload:
            self.upload_embeddings_from_dir()

    def upload_embedding_from_file(self, file_path):
        loader = TextLoader
        documents = loader(file_path).load()

        text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        docs = text_splitter.split_documents(documents)

        Chroma.from_documents(
            docs,
            OpenAIEmbeddings(),
            collection_name=self.chroma_persist_dir,
            persist_directory=self.chroma_collection_name,
        )
        print('db upload success')

    def upload_embeddings_from_dir(self):
        failed_upload_files = []

        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)

                    try:
                        self.upload_embedding_from_file(file_path)
                        print("SUCCESS: ", file_path)
                    except Exception as e:
                        print("FAILED: ", file_path + f"by({e})")
                        failed_upload_files.append(file_path)

    def query_db(self, query, use_retriever=False):
        docs = self.db.similarity_search(query) if use_retriever else self.db.as_retriever().get_relevant_documents(query)
        str_docs = "\n".join([doc.page_content for doc in docs])

        return str_docs
    
    def run_test(self):
        query = "카카오 싱크 설정"
        answer = self.query_db(query)
        print(answer)



if __name__ == "__main__":
    # db = ChromaDB()
    vdb = VectorDB()
    vdb.run_test()
