import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from utils.huggingface_uploader import HuggingFaceUploader
import tqdm
import logging
import os
import zipfile
import io
import streamlit as st

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- Custom Callback ---
class StreamlitCallback:
    def __init__(self, total_epochs, progress_bar, status_text):
        self.epoch = 0
        self.total_epochs = total_epochs
        self.progress_bar = progress_bar
        self.status_text = status_text

    def __call__(self, score, epoch, steps):
        self.epoch += 1
        progress = self.epoch / self.total_epochs
        self.progress_bar.progress(progress)
        self.status_text.text(f"Epoch {self.epoch}/{self.total_epochs} complete. Eval score: {score:.4f}")

class TrainingModel:
    def __init__(self):
        logger.debug("Initializing TrainingModel")
        # Check for available devices
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.debug("Using CUDA device")
        else:
            self.device = torch.device("cpu")
            logger.debug("Using CPU device")
            
        # Create output directory if it doesn't exist
        os.makedirs("embedding_model_output", exist_ok=True)
        
        self.dataset = None
        self.model = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.model_output_path = 'embedding_model_output'
        
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
    
    def calculate_max_steps(self, batch_size, epochs):
        """Calculate the maximum number of training steps and prepare the data loader
        
        Args:
            batch_size (int): The batch size for training
            epochs (int): The number of training epochs
            
        Returns:
            tuple: (int, DataLoader) - The maximum number of training steps and the prepared data loader
        """
        logger.debug(f"Calculating max steps with batch_size={batch_size}, epochs={epochs}")
        
        # Prepare data
        loader = self.prepare_training_data(batch_size)
        
        # Validate loader length
        if len(loader) == 0:
            logger.error("DataLoader is empty")
            raise ValueError("DataLoader is empty. Please check your dataset.")
        
        # Calculate max steps
        max_steps = len(loader) * epochs
        
        logger.debug(f"Training configuration:")
        logger.debug(f"- Number of batches per epoch: {len(loader)}")
        logger.debug(f"- Number of epochs: {epochs}")
        logger.debug(f"- Total steps: {max_steps}")
        
        return max_steps, loader
    
    def train(self, batch_size, epochs, model_id, progress_bar=None, status_text=None):
        """Train the model"""
        logger.debug(f"Starting training with batch_size={batch_size}, epochs={epochs}, model_id={model_id}")
        
        # Calculate max steps and get the loader
        max_steps, loader = self.calculate_max_steps(batch_size, epochs)
        
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
        
        # Calculate warmup steps
        warmup_steps = max(1, int(max_steps * 0.1))
        
        logger.debug(f"- Warmup steps: {warmup_steps}")
        
        # Create callback with progress tracking components
        callback = StreamlitCallback(total_epochs=epochs, progress_bar=progress_bar, status_text=status_text) if progress_bar and status_text else None
        
        # Train the model
        logger.debug("Starting model.fit")
        try:
            self.model.fit(
                train_objectives=[(loader, train_loss)],
                epochs=epochs,
                warmup_steps=warmup_steps,
                output_path=self.model_output_path,
                save_best_model=True,
                evaluator=evaluator,
                evaluation_steps=1000,
                show_progress_bar=False,
                callback=callback
            )
            st.success("Training complete!")
            logger.debug("Training completed successfully")
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
            
    def upload_to_huggingface(self, repo_name, token):
        """Upload the trained model to HuggingFace Hub"""
        if not self.model:
            raise ValueError("No trained model available to upload")
            
        try:
            logger.debug("Initializing HuggingFace uploader")
            uploader = HuggingFaceUploader(self.model_output_path)
            
            # Login to HuggingFace
            logger.debug("Logging in to HuggingFace Hub")
            uploader.login(token)
            
            # Upload model
            logger.debug(f"Uploading model to {repo_name}")
            uploader.upload_model(repo_name)
            logger.debug("Model upload completed successfully")
            
        except Exception as e:
            logger.error(f"Error uploading model to HuggingFace: {str(e)}")
            raise
        
    def evaluate(self):
        """Evaluate the model on test data"""
        if not self.model:
            raise ValueError("Model not trained yet")
            
        finetune_embeddings = HuggingFaceEmbeddings(
            model_name=self.model_output_path
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

    def get_model_download(self):
        """Create a zip file of the trained model for download
        
        Returns:
            tuple: (bytes, str) - The zip file as bytes and the filename
        """
        if not self.model or not os.path.exists(self.model_output_path):
            raise ValueError("No trained model available to download")
            
        try:
            # Create a BytesIO object to store the zip file
            zip_buffer = io.BytesIO()
            
            # Create the zip file
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Walk through the model directory
                for root, _, files in os.walk(self.model_output_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Calculate path relative to model directory for zip structure
                        arcname = os.path.relpath(file_path, self.model_output_path)
                        zip_file.write(file_path, arcname)
                        
            # Get the zip file as bytes
            zip_buffer.seek(0)
            model_name = os.path.basename(self.model_output_path)
            return zip_buffer.getvalue(), f"{model_name}.zip"
            
        except Exception as e:
            logger.error(f"Error creating model download zip: {str(e)}")
            raise 