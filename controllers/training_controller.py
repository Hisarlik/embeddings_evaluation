class TrainingController:
    def __init__(self, model, view):
        self.model = model
        self.view = view
        
    def set_datasets(self, train_data, validation_data, test_data):
        """Set the datasets in the model"""
        self.model.set_datasets(train_data, validation_data, test_data)
        
    def train(self, batch_size, epochs, model_id):
        """Train the model with given parameters"""
        self.model.train(batch_size=batch_size, epochs=epochs, model_id=model_id)
        
    def evaluate(self):
        """Evaluate the model and return results"""
        return self.model.evaluate()
        
    def upload_to_huggingface(self, repo_name, token):
        """Upload the trained model to Hugging Face Hub"""
        return self.model.upload_to_huggingface(repo_name, token) 