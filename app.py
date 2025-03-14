from flask import Flask, request, jsonify
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Dict
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

class DatasetManager:
    def __init__(self):
        self.datasets = {}
        self.embeddings_models = {
            "openai": OpenAIEmbeddings(),
            "huggingface": HuggingFaceEmbeddings()
        }
        
    def create_dataset(self, user_id: str, markdown_texts: List[str], dataset_name: str):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        documents = [Document(page_content=text) for text in markdown_texts]
        splits = text_splitter.split_documents(documents)
        
        if user_id not in self.datasets:
            self.datasets[user_id] = {}
            
        self.datasets[user_id][dataset_name] = splits
        return {"message": f"Dataset {dataset_name} created successfully", "chunks": len(splits)}

    def finetune_embeddings(self, user_id: str, dataset_name: str, input_embeddings_model: str):
        if user_id not in self.datasets or dataset_name not in self.datasets[user_id]:
            return {"error": "Dataset not found"}
            
        embeddings = self.embeddings_models.get(input_embeddings_model)
        if not embeddings:
            return {"error": "Invalid embeddings model"}
            
        documents = self.datasets[user_id][dataset_name]
        embeddings_list = embeddings.embed_documents([doc.page_content for doc in documents])
        
        return {
            "message": "Embeddings created successfully",
            "embeddings_count": len(embeddings_list)
        }

    def evaluate_retriever(self, user_id: str, dataset_name: str, query: str, input_embeddings_model: str):
        if user_id not in self.datasets or dataset_name not in self.datasets[user_id]:
            return {"error": "Dataset not found"}
            
        embeddings = self.embeddings_models.get(input_embeddings_model)
        if not embeddings:
            return {"error": "Invalid embeddings model"}
            
        query_embedding = embeddings.embed_query(query)
        # Simple cosine similarity evaluation
        documents = self.datasets[user_id][dataset_name]
        doc_embeddings = embeddings.embed_documents([doc.page_content for doc in documents])
        
        # Return top 3 most similar documents
        similarities = [
            (i, sum(a * b for a, b in zip(query_embedding, doc_emb)))
            for i, doc_emb in enumerate(doc_embeddings)
        ]
        top_results = sorted(similarities, key=lambda x: x[1], reverse=True)[:3]
        
        return {
            "results": [
                {
                    "content": documents[idx].page_content,
                    "similarity_score": score
                }
                for idx, score in top_results
            ]
        }

dataset_manager = DatasetManager()

@app.route("/dataset", methods=["POST"])
def create_dataset():
    data = request.json
    user_id = data.get("user_id")
    markdown_texts = data.get("markdown_texts")
    dataset_name = data.get("dataset_name")
    
    if not all([user_id, markdown_texts, dataset_name]):
        return jsonify({"error": "Missing required parameters"}), 400
        
    result = dataset_manager.create_dataset(user_id, markdown_texts, dataset_name)
    return jsonify(result)

@app.route("/finetune", methods=["POST"])
def finetune_embeddings():
    data = request.json
    user_id = data.get("user_id")
    dataset_name = data.get("dataset_name")
    input_embeddings_model = data.get("input_embeddings_model")
    
    if not all([user_id, dataset_name, input_embeddings_model]):
        return jsonify({"error": "Missing required parameters"}), 400
        
    result = dataset_manager.finetune_embeddings(user_id, dataset_name, input_embeddings_model)
    return jsonify(result)

@app.route("/evaluate", methods=["POST"])
def evaluate():
    data = request.json
    user_id = data.get("user_id")
    dataset_name = data.get("dataset_name")
    query = data.get("query")
    input_embeddings_model = data.get("input_embeddings_model")
    
    if not all([user_id, dataset_name, query, input_embeddings_model]):
        return jsonify({"error": "Missing required parameters"}), 400
        
    result = dataset_manager.evaluate_retriever(user_id, dataset_name, query, input_embeddings_model)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True) 