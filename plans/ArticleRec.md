# Project 2: Article Recommendation (Text Embedding/Retrieval)



**Scope/Task:**
Use the article summaries from the dataset to create text embeddings. Build a system that can take the summary of one article and recommend other articles from the dataset that have semantically similar summaries. This involves creating a simple information retrieval or semantic search system.

**Level:** 
Intermediate (Embeddings & Retrieval)

**Key Data Used:**
Article Summary

**Action Steps:**
1. [] Load article summaries from the dataset.
2.  [] Choose and load a pre-trained sentence embedding model (e.g., Sentence-BERT, or potentially use the text encoder part of a fine-tuned BERT from Project 1 if applicable).
3.  [] Generate embeddings for all article summaries in the dataset.
4.  [] Build an index for the embeddings. This could be a simple in-memory list for small datasets or use a library like FAISS for larger datasets and faster similarity search.
5.  [] Implement a query function: given a summary (or its embedding), calculate its similarity to all other embeddings in the index.
6.  [] Retrieve the top-N most similar article embeddings and display the corresponding article titles/summaries as recommendations.
    * [] Update this with a RAG-like system!

**Key Learning Outcomes:**
-   Understanding concepts of text embeddings and vector representations of text.
-   Using libraries for generating sentence-level embeddings.
-   Concepts of semantic similarity and distance metrics (e.g., cosine similarity).
-   Designing and implementing a basic information retrieval or recommendation system based on content similarity.
-   Basic introduction to vector indexing concepts.

**Estimated GPU Needs:**
GPU is helpful for the embedding generation step (e.g., 8GB VRAM), especially for larger datasets or more complex embedding models.

**Key Resources/Tools:**
-   Hugging Face `transformers` (for text models)
-  `sentence-transformers` library (specifically designed for sentence embeddings)
-   FAISS (optional, for efficient similarity search on larger datasets)
-   numpy and scipy (for vector operations and similarity calculations if not using FAISS)
-   News 24 dataset