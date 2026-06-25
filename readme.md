# Exploring Retrieval Augmented Generation (RAG) Techniques

This Google Colab notebook serves as an in-depth exploration of various Retrieval Augmented Generation (RAG) architectures and their effectiveness in question-answering tasks.

## Project Overview

Retrieval Augmented Generation (RAG) combines the power of large language models (LLMs) with external knowledge bases to provide more accurate, up-to-date, and contextually relevant answers. This notebook systematically builds and compares different RAG approaches, from basic semantic search to advanced hybrid methods incorporating contextualization and reranking.

## Approaches Explored

1.  **Naive RAG (Semantic Search)**:
    *   Utilizes OpenAI embeddings (`text-embedding-3-small`) for semantic similarity search.
    *   Chunks documents using a recursive chunking strategy.
    *   Stores embeddings in ChromaDB.
    *   Answers questions by fetching top-K semantically similar chunks and passing them to an LLM.

2.  **BM25 Keyword Search**:
    *   Implements BM25Okapi for keyword-based search.
    *   Complements semantic search by capturing exact term matches.

3.  **Hybrid Search with Reciprocal Rank Fusion (RRF)**:
    *   Combines the strengths of semantic and keyword search.
    *   Uses Reciprocal Rank Fusion (RRF) to merge and re-rank results from both methods, offering a more robust retrieval.

4.  **Contextualized Chunking (Anthropic Claude)**:
    *   Enhances chunk embeddings by prepending LLM-generated summaries (using Anthropic Claude) that describe the chunk's context within its original document.
    *   This aims to improve the quality of embeddings by adding more descriptive metadata.

5.  **Reranking (Cross-Encoder)**:
    *   Introduces a `CrossEncoder` model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to re-score the top retrieved documents.
    *   Reranking considers the query and retrieved document pair together, leading to more precise relevance scoring.

6.  **Full RAG Pipeline**:
    *   Integrates contextualized chunking, hybrid search, and cross-encoder reranking into a comprehensive RAG system.
    *   Compares its performance against simpler RAG setups.

## Technologies Used

*   **Python Libraries**: `openai`, `chromadb`, `rank_bm25`, `sentence-transformers`, `pypdf`, `tiktoken`, `anthropic`.
*   **LLMs**: OpenAI's GPT-4o-mini for generation, Anthropic's Claude for contextualization.
*   **Vector Database**: ChromaDB.
*   **File Processing**: `pypdf` for loading documents from PDF files.

## Getting Started

1.  **Clone the Repository**:
    ```bash
    git clone <your-repo-link>
    cd <your-repo-name>
    ```
2.  **Open in Google Colab**:
    Upload the `.ipynb` file to Google Colab.
3.  **Install Dependencies**:
    Run the first cell to install all necessary Python packages.
4.  **Configure API Keys**:
    Add your OpenAI and Anthropic API keys to Colab's `Secrets` manager (accessible via the 🔑 icon in the left panel) with the names `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` respectively.
5.  **Run Cells Sequentially**:
    Execute the notebook cells in order to see each RAG component in action and observe the comparisons.

## Insights & Comparisons

This notebook provides side-by-side comparisons of different RAG approaches to demonstrate how each enhancement (hybrid search, contextualization, reranking) contributes to the overall effectiveness and accuracy of the generated answers. Pay close attention to the retrieved chunks and the final LLM responses for each query across different RAG variants.

## Future Work

*   Experiment with different chunking strategies and parameters.
*   Explore other embedding models and rerankers.
*   Implement query rewriting or decomposition for complex questions.
*   Integrate evaluation metrics for quantitative comparison of RAG pipelines.
