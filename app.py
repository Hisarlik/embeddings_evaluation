import streamlit as st
import json
import logging
import os
import torch
import base64
import time
from pathlib import Path


def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Load the logo image
logo_path = Path("static/gao-techhub-logo.png")
encoded_logo = get_base64_encoded_image(str(logo_path))

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

# or simply:
torch.classes.__path__ = []

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config first - must be the first Streamlit command
st.set_page_config(
    page_title="NTT DATA GAO Tech Hub - Embeddings Training",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'model_initialized' not in st.session_state:
    st.session_state.model_initialized = False
    
if 'components' not in st.session_state:
    st.session_state.components = None

# Load external CSS
def load_css():
    with open("static/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

@st.cache_resource
def init_components():
    """Initialize MVC components with error handling and caching"""
    try:
        # Import components here to avoid circular imports
        from models.training_model import TrainingModel
        from views.training_view import TrainingView
        from controllers.training_controller import TrainingController
        
        # Create output directories
        os.makedirs("models/snowflake", exist_ok=True)
        
        model = TrainingModel()
        view = TrainingView()
        controller = TrainingController(model, view)
        
        st.session_state.model_initialized = True
        st.session_state.components = (model, view, controller)
        
        return model, view, controller
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        st.error(f"Failed to initialize application components: {str(e)}")
        return None, None, None

def handle_file_upload(file):
    """Handle file upload with error handling"""
    if file is not None:
        try:
            return json.load(file)
        except json.JSONDecodeError:
            st.error(f"Invalid JSON format in file: {file.name}")
            return None
        except Exception as e:
            st.error(f"Error processing file {file.name}: {str(e)}")
            return None
    return None

def main():
    try:
        # Add company header
        st.markdown(f"""
            <div class="company-header">
                <img src="data:image/png;base64,{encoded_logo}" alt="GAO TechHub Logo" class="company-logo">
            </div>
        """, unsafe_allow_html=True)

        # Main content
        st.markdown("""
            <div style="text-align: center; margin-bottom: 3rem;">
                <h1 style="font-size: 2.5rem; margin-bottom: 1rem;">Embeddings Training App</h1>
                <p style="color: var(--text-secondary); font-size: 1.1rem;">Upload your datasets and train your embeddings model</p>
            </div>
        """, unsafe_allow_html=True)

        # Initialize components if not already initialized
        if not st.session_state.model_initialized:
            model, view, controller = init_components()
            if None in (model, view, controller):
                st.stop()
        else:
            model, view, controller = st.session_state.components

        # File uploaders in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="upload-header">Training Dataset</div>', unsafe_allow_html=True)
            train_file = st.file_uploader("Drag and drop file here", type=['json'], key="train")
            st.markdown('<p style="color: var(--text-secondary); font-size: 0.75rem;">Limit 200MB per file • JSON</p>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="upload-header">Validation Dataset</div>', unsafe_allow_html=True)
            validation_file = st.file_uploader("Drag and drop file here", type=['json'], key="validation")
            st.markdown('<p style="color: var(--text-secondary); font-size: 0.75rem;">Limit 200MB per file • JSON</p>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="upload-header">Test Dataset</div>', unsafe_allow_html=True)
            test_file = st.file_uploader("Drag and drop file here", type=['json'], key="test")
            st.markdown('<p style="color: var(--text-secondary); font-size: 0.75rem;">Limit 200MB per file • JSON</p>', unsafe_allow_html=True)

        if all([train_file, validation_file, test_file]):
            # Process uploaded files with error handling
            train_data = handle_file_upload(train_file)
            validation_data = handle_file_upload(validation_file)
            test_data = handle_file_upload(test_file)
            
            if all([train_data, validation_data, test_data]):
                try:
                    controller.set_datasets(train_data, validation_data, test_data)
                    
                    # Training parameters section
                    st.markdown("""
                        <div class="upload-container">
                            <h2 style="font-size: 1.5rem; margin-bottom: 1rem;">Training Parameters</h2>
                            <p class="upload-description">Configure your model training parameters</p>
                            <p class="upload-description">Select the base model for fine-tuning</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        batch_size = st.number_input("Batch Size", min_value=1, value=2)
                    with col2:
                        epochs = st.number_input("Number of Epochs", min_value=1, value=1)
                    with col3:
                        model_options = [
                            "Snowflake/snowflake-arctic-embed-m",
                            "Snowflake/snowflake-arctic-embed-m-v2.0",
                            "Alibaba-NLP/gte-modernbert-base",
                            "nreimers/albert-small-v2"
                        ]
                        model_id = st.selectbox("Model ID", model_options)
                    
                    # Add HuggingFace upload section
                    st.markdown("""
                        <div class="upload-container">
                            <h2 style="font-size: 1.5rem; margin-bottom: 1rem;">HuggingFace Upload (Optional)</h2>
                            <p class="upload-description">Configure HuggingFace upload settings</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        hf_repo_name = st.text_input("HuggingFace Repository Name", placeholder="username/model-name", help="Format: username/model-name")
                    with col2:
                        hf_token = st.text_input("HuggingFace Token", type="password", help="Your HuggingFace API token")
                    
                    if st.button("Start Training"):
                        with st.spinner("Training in progress..."):
                            try:
                                # Delete embedding_model_output folder if it exists
                                if os.path.exists("embedding_model_output"):
                                    import shutil
                                    shutil.rmtree("embedding_model_output")
                                
                                # Create progress tracking components
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                   
                                controller.train(
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    model_id=model_id,
                                    hf_repo_name=hf_repo_name if hf_repo_name else None,
                                    hf_token=hf_token if hf_token else None,
                                    progress_bar=progress_bar,
                                    status_text=status_text
                                )
                                time.sleep(1)
                                progress_bar.empty()
                                status_text.empty()
                                
                                # Show evaluation results
                                st.markdown("""
                                    <div class="upload-container">
                                        <h2 style="font-size: 1.5rem; margin-bottom: 1rem;">Evaluation Results</h2>
                                    </div>
                                """, unsafe_allow_html=True)
                                results = controller.evaluate()
                                view.display_results(results, model=controller.model)
                            except Exception as e:
                                logger.error(f"Training error: {str(e)}")
                                st.error("An error occurred during training. Please check the logs for details.")
                                
                except Exception as e:
                    logger.error(f"Error setting datasets: {str(e)}")
                    st.error("Failed to process the uploaded datasets. Please check the file format and try again.")

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An unexpected error occurred. Please refresh the page and try again.")

if __name__ == "__main__":
    main() 