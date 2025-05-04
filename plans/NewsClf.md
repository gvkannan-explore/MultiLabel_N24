# Project 1: News Category Classification (Text Only)

**Scope/Task:**
Fine-tune a BERT-based model on the article text (headline + body) to classify the news category. This is a standard text classification task using a pre-trained transformer model with RoBERTa or DistilBERT.

**Level:**
Beginner


**Key Data Used:**
N24 Dataset - Article Text (Headline + Body), Category Label

**Action Steps:**
- [] Preprocess text data & create data-module (cleaning, tokenization).
    - [] Decide which categories to exclude if the dataset is too large for the scope of the project.
    - [] Split data into training, validation, and test sets.
- [] Create a baseline model
    - [] Load a pre-trained BERT model and its corresponding tokenizer from Hugging Face to evaluate baseline training conditions.
    - [] Decide if other variants of BERT can be tested against

- [] Configure fine-tuning parameters (learning rate, epochs, batch size).
- [] Train the model and dockerize the training pipeline for repeatibility.
- [] Evaluation & testing.
- [] Test against an existing open source or open-weights LLM (Llama3-7B, Qwen-7B, etc.)


**Key Outcomes:**
- Practical experience using the Hugging Face `transformers` and `datasets` libraries for a common NLP task.
-   Text preprocessing and tokenization techniques for transformer models.
-   Setting up and executing a fine-tuning process for classification.
-   Evaluating the performance of NLP models.


**Estimated GPU Needs:**
Modest GPU (e.g., 8GB VRAM) is highly recommended for faster training. Can potentially run on CPU for inference. Specific instructions shall be provided later!

**Key Resources/Tools:**
-   Hugging Face `transformers` & `sentence-transformers` library (for models and tokenizers)
-   Hugging Face `datasets` library (for data loading and management)
-   PyTorch (as the backend framework)
-   scikit-learn (for evaluation metrics)
-   News 24 dataset