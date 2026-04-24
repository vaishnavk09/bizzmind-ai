import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

class VectorStoreManager:
    """
    Manages FAISS vector store using HuggingFace embeddings.
    """
    def __init__(self, index_path="faiss_index"):
        self.index_path = index_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = None

    def build_store(self, text_chunks: list):
        """
        Creates FAISS index from text chunks and saves it locally.
        """
        print(f"Building FAISS vector store with {len(text_chunks)} chunks...")
        self.vector_store = FAISS.from_texts(text_chunks, self.embeddings)
        
        # Save index locally
        os.makedirs(self.index_path, exist_ok=True)
        self.vector_store.save_local(self.index_path)
        print(f"Vector store saved to {self.index_path}")

    def load_store(self):
        """
        Reloads existing index from local storage.
        """
        if os.path.exists(self.index_path):
            try:
                self.vector_store = FAISS.load_local(
                    self.index_path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                print("Vector store loaded successfully.")
            except Exception as e:
                print(f"Error loading vector store: {e}")
        else:
            print(f"Index path {self.index_path} does not exist.")

    def similarity_search(self, query: str, k: int = 5) -> list:
        """
        Returns top-k relevant chunks for the query.
        """
        if not self.vector_store:
            print("Vector store not initialized. Load or build it first.")
            return []
        
        results = self.vector_store.similarity_search(query, k=k)
        return [res.page_content for res in results]

if __name__ == "__main__":
    # Test block
    from ingestion import DataIngestionPipeline
    
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "mock_store.csv")
    pipeline = DataIngestionPipeline()
    if os.path.exists(csv_path):
        df = pipeline.load_csv(csv_path)
        df = pipeline.clean_data(df)
        df = pipeline.add_features(df)
        chunks = pipeline.to_text_chunks(df)
        
        vsm = VectorStoreManager(index_path=os.path.join(os.path.dirname(__file__), "..", "faiss_index"))
        vsm.build_store(chunks)
        
        query = "Who bought Rice recently?"
        print(f"\nTesting similarity search for: '{query}'")
        results = vsm.similarity_search(query, k=3)
        for i, res in enumerate(results):
            print(f"Result {i+1}: {res}")
    else:
        print("Run generate_mock.py to create the dataset first.")
