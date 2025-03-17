# Embeddings Training App

A Streamlit application for training and evaluating embeddings models using the MVC pattern.

## Features

- Upload custom training datasets in JSON format
- Configure training parameters (batch size, epochs, model)
- Train embeddings models with Matryoshka loss
- Evaluate model performance with detailed metrics
- Visualize results with interactive plots

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Upload your training dataset in JSON format with the following structure:
```json
{
    "train": {
        "corpus": {...},
        "questions": {...},
        "relevant_contexts": {...}
    },
    "validation": {
        "corpus": {...},
        "questions": {...},
        "relevant_contexts": {...}
    },
    "test": {
        "corpus": {...},
        "questions": {...},
        "relevant_contexts": {...}
    }
}
```

3. Configure training parameters and click "Start Training"

4. View the evaluation results and metrics

## Project Structure

- `app.py`: Main Streamlit application
- `models/training_model.py`: Model class handling training logic
- `views/training_view.py`: View class for displaying results
- `controllers/training_controller.py`: Controller coordinating model and view
- `requirements.txt`: Project dependencies 