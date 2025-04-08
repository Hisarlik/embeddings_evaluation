import os
import logging
from huggingface_hub import HfApi, login
import shutil

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class HuggingFaceUploader:
    def __init__(self, model_path):
        """
        Initialize the HuggingFaceUploader
        
        Args:
            model_path (str): Path to the trained model directory
        """
        self.model_path = model_path
        self.api = None
        
    def login(self, token=None):
        """
        Login to Hugging Face Hub
        
        Args:
            token (str, optional): Hugging Face token. If not provided, will look for HUGGINGFACE_TOKEN environment variable.
            
        Raises:
            ValueError: If no token is provided and HUGGINGFACE_TOKEN environment variable is not set
        """
        if not token:
            token = os.getenv("HUGGINGFACE_TOKEN")
            if not token:
                raise ValueError("Hugging Face token not provided. Please set HUGGINGFACE_TOKEN environment variable or provide token directly.")
        
        try:
            login(token=token)
            self.api = HfApi()
            logger.debug("Successfully logged in to Hugging Face Hub")
        except Exception as e:
            logger.error(f"Error logging in to Hugging Face Hub: {str(e)}")
            raise
            
    def upload_model(self, repo_name):
        """
        Upload the model to Hugging Face Hub
        
        Args:
            repo_name (str): Name of the repository (e.g., username/model-name)
            
        Returns:
            bool: True if upload was successful
            
        Raises:
            ValueError: If model path does not exist
            Exception: If upload fails
        """
        if not os.path.exists(self.model_path):
            raise ValueError(f"Model path does not exist: {self.model_path}")
            
        if not self.api:
            raise ValueError("Not logged in to Hugging Face Hub. Call login() first.")
            
        try:
            # Create repository if it doesn't exist
            try:
                self.api.create_repo(repo_name, repo_type="model", exist_ok=True)
            except Exception as e:
                logger.warning(f"Repository creation warning: {str(e)}")
            
            # Save model to temporary directory
            temp_path = "temp_model"
            shutil.copytree(self.model_path, temp_path)
            
            # Upload model files
            logger.debug("Uploading model files to Hugging Face Hub")
            self.api.upload_folder(
                folder_path=temp_path,
                repo_id=repo_name,
                repo_type="model",
                ignore_patterns=["*.pyc", "__pycache__", "*.pyo", "*.pyd", ".Python", "build", "develop-eggs", "dist", "downloads", "eggs", ".eggs", "lib", "lib64", "parts", "sdist", "var", "wheels", "*.egg-info", ".installed.cfg", "*.egg"]
            )
            
            # Clean up temporary directory
            shutil.rmtree(temp_path)
            
            logger.debug(f"Model successfully uploaded to {repo_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading model to Hugging Face Hub: {str(e)}")
            raise 