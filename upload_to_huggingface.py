import os
import argparse
from utils.huggingface_uploader import HuggingFaceUploader

def main():
    parser = argparse.ArgumentParser(description='Upload trained model to Hugging Face Hub')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model directory')
    parser.add_argument('--repo_name', type=str, required=True,
                      help='Name of the Hugging Face repository (e.g., username/model-name)')
    parser.add_argument('--token', type=str,
                      help='Hugging Face token (optional, can also be set via HUGGINGFACE_TOKEN env variable)')
    
    args = parser.parse_args()
    
    try:
        # Initialize uploader
        uploader = HuggingFaceUploader(args.model_path)
        
        # Login to Hugging Face
        uploader.login(args.token)
        
        # Upload model
        uploader.upload_model(args.repo_name)
        
        print(f"Model successfully uploaded to {args.repo_name}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main() 