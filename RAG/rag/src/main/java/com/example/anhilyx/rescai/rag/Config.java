package com.example.anhilyx.rescai.rag;

/**
 * Centralized configuration for the RAG system.
 */
public class Config {

    public static final int N_RESULTS = 5;  // The number of results to return for a query.

    public static final String TOKENIZER_FILE = "tokenizer.json";
    public static final String MODEL_FILE = "model.onnx";

    public static final String REPO_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2";
    public static final int EMBEDDING_DIM = 384;  // The dimensionality of the embedding vectors produced by the model.

    public static final int N_TOKENS = 128;  // The maximum number of tokens.
    public static final int CHUNK_SIZE = 1024;  // The minimum number of characters in each chunk.
}
