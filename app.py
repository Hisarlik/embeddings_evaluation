import streamlit as st
import json
from models.training_model import TrainingModel
from views.training_view import TrainingView
from controllers.training_controller import TrainingController

# Enable debug mode
#st.set_option('deprecation.showPyplotGlobalUse', False)
#st.set_option('deprecation.showfileUploaderEncoding', False)

def main():
    st.title("Embeddings Training App")
    st.write("Upload your training, validation, and test datasets")

    # Initialize MVC components
    model = TrainingModel()
    view = TrainingView()
    controller = TrainingController(model, view)

    # File uploaders for each dataset
    col1, col2, col3 = st.columns(3)
    
    with col1:
        train_file = st.file_uploader("Training Dataset (JSON)", type=['json'])
    with col2:
        validation_file = st.file_uploader("Validation Dataset (JSON)", type=['json'])
    with col3:
        test_file = st.file_uploader("Test Dataset (JSON)", type=['json'])
    
    if train_file is not None and validation_file is not None and test_file is not None:
        try:
            train_data = json.load(train_file)
            validation_data = json.load(validation_file)
            test_data = json.load(test_file)
            
            controller.set_datasets(train_data, validation_data, test_data)
            
            # Training parameters
            st.subheader("Training Parameters")
            batch_size = st.number_input("Batch Size", min_value=1, value=2)
            epochs = st.number_input("Number of Epochs", min_value=1, value=1)
            model_id = st.text_input("Model ID", value="Snowflake/snowflake-arctic-embed-m")
            
            if st.button("Start Training"):
                with st.spinner("Training in progress..."):
                    controller.train(
                        batch_size=batch_size,
                        epochs=epochs,
                        model_id=model_id
                    )
                    
                    # Show evaluation results
                    st.subheader("Evaluation Results")
                    results = controller.evaluate()
                    view.display_results(results)
                    
        except json.JSONDecodeError:
            st.error("Invalid JSON file. Please upload valid JSON files.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)  # This will show the full traceback

if __name__ == "__main__":
    main() 