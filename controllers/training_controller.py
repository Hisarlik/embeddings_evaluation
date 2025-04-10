class TrainingController:
    def __init__(self, model, view):
        self.model = model
        self.view = view
        
    def set_datasets(self, train_data, validation_data, test_data):
        """Set the datasets in the model"""
        self.model.set_datasets(train_data, validation_data, test_data)
        
    def train(self, batch_size, epochs, model_id, hf_repo_name=None, hf_token=None, progress_bar=None, status_text=None):
        """Train the model with given parameters and optionally upload to HuggingFace"""
        # Train the model
        self.model.train(batch_size=batch_size, epochs=epochs, model_id=model_id, 
                        progress_bar=progress_bar, status_text=status_text)
        
        # Upload to HuggingFace if credentials provided
        if hf_repo_name and hf_token:
            self.model.upload_to_huggingface(hf_repo_name, hf_token)
        
    def evaluate(self):
        """Evaluate the model and return results"""
        return self.model.evaluate() 