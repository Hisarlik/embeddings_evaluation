import json
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sentence_transformers import InputExample
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import pandas as pd
import tqdm
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document

# Device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
else:
    device = torch.device("cpu")
    print("MPS device not found, using CPU")

def load_json_file(file_path):
    """
    Load and return data from a JSON file
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        dict: Loaded JSON data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def evaluate(
    dataset,
    embed_model,
    top_k=1,
    verbose=False,
):
  corpus = dataset['corpus']
  questions = dataset['questions']
  relevant_docs = dataset['relevant_contexts']

  relevant_docs_filtrado = {clave: valor for clave, valor in relevant_docs.items() if clave in questions}

  documents = [Document(page_content=content, metadata={"id": doc_id}) for doc_id, content in corpus.items()]
  vectorstore = FAISS.from_documents(documents, embed_model)

  retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

  eval_results = []
  for id, question in tqdm.tqdm(questions.items()):
    retrieved_nodes = retriever.invoke(question)
    retrieved_ids = [node.metadata["id"] for node in retrieved_nodes]
    expected_id = relevant_docs[id][0]
    is_hit = expected_id in retrieved_ids
    eval_results.append({"id": id, "question": question, "expected_id": expected_id, "is_hit": is_hit})

  return eval_results

# Define data directory path
data_dir = "data_clean"  # Relative path from the project root

# configure embeddings training batch size 
BATCH_SIZE = 2

# Embeddings base model to finetuning
model_id = "Snowflake/snowflake-arctic-embed-m"

# Number of epochs
EPOCHS = 1


# Load train, validation and test datasets
train_data = load_json_file(os.path.join(data_dir, "train_dataset_clean.json"))
val_data = load_json_file(os.path.join(data_dir, "validation_dataset_clean.json"))
test_data = load_json_file(os.path.join(data_dir, "test_dataset_clean.json"))




train_corpus = train_data['corpus']
train_queries = train_data['questions']
train_relevant_docs = train_data['relevant_contexts']

examples = []
for query_id, query in train_queries.items():
    try:
        doc_id = train_relevant_docs[query_id][0]
        text = train_corpus[doc_id]
        example = InputExample(texts=[query, text])
        examples.append(example)
        
            
    except:
        print(f"Error processing query_id: {query_id}")


# Loader for the training data
loader = DataLoader(
    examples, batch_size=BATCH_SIZE
)

#Loading base model
model = SentenceTransformer(model_id, trust_remote_code=True)
model = model.to(device)

# Define loss function
matryoshka_dimensions = [768, 512, 256, 128, 64]
inner_train_loss = MultipleNegativesRankingLoss(model)
train_loss = MatryoshkaLoss(
    model, inner_train_loss, matryoshka_dims=matryoshka_dimensions
)

# Define validation data
val_corpus = val_data['corpus']
val_queries = val_data['questions']
val_relevant_docs = val_data['relevant_contexts']

relevant_docs_filtered= {clave: valor for clave, valor in val_relevant_docs.items() if clave in val_queries}

evaluator = InformationRetrievalEvaluator(val_queries, val_corpus, relevant_docs_filtered)

#TODO: analyze this
valores_relevant_docs = set(tuple(valor) for valor in val_relevant_docs.values())


warmup_steps = int(len(loader) * EPOCHS * 0.1)

# Finetune embeddings
model.fit(
    train_objectives=[(loader, train_loss)],
    epochs=EPOCHS,
    warmup_steps=warmup_steps,
    output_path='models/snowflake/finetuned_snowflake_clean',
    checkpoint_path="models/snowflake/finetuned_snowflake_clean_checkpoint",  # Guarda checkpoints aquí
    checkpoint_save_steps=500,  # Guarda cada 50 pasos de evaluación
    checkpoint_save_total_limit=3,  # Mantiene solo los 3 checkpoints más recientes    
    show_progress_bar=True,
    evaluator=evaluator,
    evaluation_steps=100
    # use_amp=True
)

# Evaluate finetuned embeddings
finetune_embeddings = HuggingFaceEmbeddings(model_name="models/snowflake/finetuned_snowflake_clean")
finetune_results = evaluate(test_data, finetune_embeddings)

# Convert results to DataFrame and calculate hit rate
finetune_results_df = pd.DataFrame(finetune_results)
finetune_hit_rate = finetune_results_df["is_hit"].mean()
finetune_hit_rate



