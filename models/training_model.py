import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from huggingface_hub import HfApi, login
import pandas as pd
import tqdm
import logging
import os
from utils.huggingface_uploader import HuggingFaceUploader

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TrainingModel:
    def __init__(self):
        logger.debug("Initializing TrainingModel")
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        logger.debug(f"Using device: {self.device}")
        self.dataset = None
        self.model = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
    def set_datasets(self, train_data, validation_data, test_data):
        """Set the datasets for training, validation and testing"""
        logger.debug("Setting datasets")
        self.train_data = train_data
        self.val_data = validation_data
        self.test_data = test_data
        logger.debug(f"Dataset sizes - Train: {len(self.train_data.get('corpus', {}))}, Val: {len(self.val_data.get('corpus', {}))}, Test: {len(self.test_data.get('corpus', {}))}")
        
    def prepare_training_data(self, batch_size):
        """Prepare training data and create DataLoader"""
        logger.debug("Preparing training data")
        examples = []
        train_corpus = self.train_data.get('corpus', {})
        train_queries = self.train_data.get('questions', {})
        train_relevant_docs = self.train_data.get('relevant_contexts', {})
        
        # Validate that we have data
        if not train_corpus or not train_queries or not train_relevant_docs:
            logger.error("Training data is empty")
            raise ValueError("Training data is empty. Please check your dataset structure.")
        
        logger.debug(f"Found {len(train_queries)} queries and {len(train_corpus)} documents")
        
        for query_id, query in train_queries.items():
            try:
                doc_id = train_relevant_docs[query_id][0]
                text = train_corpus[doc_id]
                example = InputExample(texts=[query, text])
                examples.append(example)
            except Exception as e:
                logger.error(f"Error processing query_id: {query_id}. Error: {str(e)}")
        
        # Validate that we have examples
        if not examples:
            logger.error("No valid training examples were created")
            raise ValueError("No valid training examples were created. Please check your dataset.")
            
        logger.debug(f"Created {len(examples)} training examples")
        return DataLoader(examples, batch_size=batch_size, shuffle=True)
    
    def train(self, batch_size, epochs, model_id):
        """Train the model"""
        logger.debug(f"Starting training with batch_size={batch_size}, epochs={epochs}, model_id={model_id}")
        
        # Prepare data
        loader = self.prepare_training_data(batch_size)
        
        # Validate loader length
        if len(loader) == 0:
            logger.error("DataLoader is empty")
            raise ValueError("DataLoader is empty. Please check your dataset.")
        
        # Initialize model
        logger.debug("Initializing model")
        self.model = SentenceTransformer(model_id, trust_remote_code=True)
        self.model = self.model.to(self.device)
        logger.debug("Model initialized and moved to device")
        
        # Define loss function
        logger.debug("Setting up loss functions")
        matryoshka_dimensions = [768, 512, 256, 128, 64]
        inner_train_loss = MultipleNegativesRankingLoss(self.model)
        train_loss = MatryoshkaLoss(
            self.model, inner_train_loss, matryoshka_dims=matryoshka_dimensions
        )
        
        # Prepare validation data
        logger.debug("Preparing validation data")
        val_corpus = self.val_data.get('corpus', {})
        val_queries = self.val_data.get('questions', {})
        val_relevant_docs = self.val_data.get('relevant_contexts', {})
        relevant_docs_filtered = {k: v for k, v in val_relevant_docs.items() if k in val_queries}
        
        evaluator = InformationRetrievalEvaluator(val_queries, val_corpus, relevant_docs_filtered)
        
        # Calculate warmup steps and max steps
        warmup_steps = max(1, int(len(loader) * epochs * 0.1))
        max_steps = len(loader) * epochs
        
        logger.debug(f"Training configuration:")
        logger.debug(f"- Number of batches per epoch: {len(loader)}")
        logger.debug(f"- Number of epochs: {epochs}")
        logger.debug(f"- Total steps: {max_steps}")
        logger.debug(f"- Warmup steps: {warmup_steps}")
        
        # Train the model
        logger.debug("Starting model.fit")
        try:
            self.model.fit(
                train_objectives=[(loader, train_loss)],
                epochs=epochs,
                warmup_steps=warmup_steps,
                output_path='models/snowflake/finetuned_snowflake_clean',
                checkpoint_path="models/snowflake/finetuned_snowflake_clean_checkpoint",
                checkpoint_save_steps=500,
                checkpoint_save_total_limit=3,
                show_progress_bar=True,
                evaluator=evaluator,
                evaluation_steps=100
            )
            logger.debug("Training completed successfully")
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
        
    def evaluate(self):
        """Evaluate the model on test data"""
        if not self.model:
            raise ValueError("Model not trained yet")
            
        finetune_embeddings = HuggingFaceEmbeddings(
            model_name="models/snowflake/finetuned_snowflake_clean"
        )
        
        corpus = self.test_data.get('corpus', {})
        questions = self.test_data.get('questions', {})
        relevant_docs = self.test_data.get('relevant_contexts', {})
        
        relevant_docs_filtered = {k: v for k, v in relevant_docs.items() if k in questions}
        
        documents = [
            Document(page_content=content, metadata={"id": doc_id})
            for doc_id, content in corpus.items()
        ]
        
        vectorstore = FAISS.from_documents(documents, finetune_embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
        
        eval_results = []
        for id, question in tqdm.tqdm(questions.items()):
            retrieved_nodes = retriever.invoke(question)
            retrieved_ids = [node.metadata["id"] for node in retrieved_nodes]
            expected_id = relevant_docs[id][0]
            is_hit = expected_id in retrieved_ids
            eval_results.append({
                "id": id,
                "question": question,
                "expected_id": expected_id,
                "is_hit": is_hit
            })
            
        return eval_results 

    def upload_to_huggingface(self, repo_name, token=None):
        """Upload the trained model to Hugging Face Hub"""
        logger.debug(f"Preparing to upload model to Hugging Face Hub: {repo_name}")
        
        if not self.model:
            raise ValueError("Model not trained yet")
            
        try:
            # Initialize uploader with the model path
            uploader = HuggingFaceUploader("models/snowflake/finetuned_snowflake_clean")
            
            # Login and upload
            uploader.login(token)
            return uploader.upload_model(repo_name)
            
        except Exception as e:
            logger.error(f"Error uploading model to Hugging Face Hub: {str(e)}")
            raise 