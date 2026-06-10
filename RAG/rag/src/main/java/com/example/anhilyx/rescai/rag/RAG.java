package com.example.anhilyx.rescai.rag;

import android.content.Context;

import com.tom_roush.pdfbox.android.PDFBoxResourceLoader;

import java.io.File;
import java.io.InputStream;
import java.io.ObjectInputFilter.Config;

import ai.onnxruntime.OrtException;

/**
 * Main class for the RAG (Retrieval-Augmented Generation) system.
 * This class provides methods to initialize the RAG, create a RAG index from a PDF file and query it.
 */
public class RAG {

    protected static Tokenizer tokenizer;
    protected static EmbeddingEngine embeddingEngine;
    protected static ObjectBox objectBox;

    /**
     * Callback interface for reporting progress and handling success or error during RAG operations.
     */
    public interface RAGCallback {

        public static final int STEP_INITIALIZING = 100;
        public static final int STEP_INSTALLING_MODEL = 101;

        public static final int STEP_READING_PDF = 200;
        public static final int STEP_CHUNKING_PDF = 201;
        public static final int STEP_CREATING_RAG = 202;

        /**
         * Called when the operation completes successfully.
         */
        public void onSuccess();

        /**
         * Called when an error occurs during the operation.
         * @param e The exception that was thrown.
         * @param step The step of the operation during which the error occurred (e.g., STEP_INITIALIZING, STEP_READING_PDF, etc.).
         */
        public void onError(Exception e, int step);

        /**
         * Called to report progress during the operation.
         * @param progress A float value between 0.0 and 1.0 indicating the progress of the operation.
         * @param step The step of the operation during which the progress is being reported (e.g., STEP_INITIALIZING, STEP_READING_PDF, etc.).
         */
        public void onProgress(float progress, int step);
    }

    /**
     * Initialize the RAG system by loading the tokenizer, embedding engine, and ObjectBox database.
     * @param context The Android context, used to access the app's file directory for loading the tokenizer and model files.
     * @param callback A callback interface to report progress and handle success or error during initialization.
     * @param repo The repository ID to download the model from.
     */
    public static void init(Context context, RAGCallback callback, String repo) {

        // Check if the model file already exists to avoid unnecessary reinstallation
        if (
                !new File(context.getFilesDir(), Config.MODEL_FILE).exists() ||
                repo != null
        ) {
            // Install the model
            HFDownloader.downloadRepository(context.getFilesDir(), repo != null ? repo : Config.REPO_ID, new Downloader.DownloadCallback() {
                @Override
                public void onSuccess() {
                    _init2(context, callback);
                }

                @Override
                public void onError(int code, String message) {
                    callback.onError(new Exception("Model download failed with code " + code + ": " + message), RAGCallback.STEP_INSTALLING_MODEL);
                }

                @Override
                public void onProgress(float progress) {
                    callback.onProgress(progress, RAGCallback.STEP_INSTALLING_MODEL);
                }
            });
        }

        else {
            _init2(context, callback);
        }
    }

    /**
     * Initialize the RAG system by loading the tokenizer, embedding engine, and ObjectBox database.
     * @param context The Android context, used to access the app's file directory for loading the tokenizer and model files.
     * @param callback A callback interface to report progress and handle success or error during initialization.
     */
    public static void init(Context context, RAGCallback callback) {
        init(context, callback, null);
    }

    /**
     * Helper method to initialize the RAG system after the model has been installed (or if it already exists).
     * @param context The Android context, used to access the app's file directory for loading the tokenizer and model files.
     * @param callback A callback interface to report progress and handle success or error during initialization.
     */
    private static void _init2(Context context, RAGCallback callback) {

        // Initialize the elements of the RAG
        try {
            callback.onProgress(0.0f, RAGCallback.STEP_INITIALIZING);

            tokenizer = new Tokenizer(new File(context.getFilesDir(), Config.TOKENIZER_FILE));
            callback.onProgress(0.2f, RAGCallback.STEP_INITIALIZING);

            embeddingEngine = new EmbeddingEngine(new File(context.getFilesDir(), Config.MODEL_FILE));
            callback.onProgress(0.4f, RAGCallback.STEP_INITIALIZING);

            ObjectBox.init(context);
            callback.onProgress(0.6f, RAGCallback.STEP_INITIALIZING);

            objectBox = new ObjectBox();
            callback.onProgress(0.8f, RAGCallback.STEP_INITIALIZING);

            PDFBoxResourceLoader.init(context.getApplicationContext());
            callback.onProgress(1.0f, RAGCallback.STEP_INITIALIZING);

        } catch (Exception e) {
            callback.onError(e, RAGCallback.STEP_INITIALIZING);
        }

        callback.onSuccess();
    }

    /**
     * Create a RAG index from the given PDF file.
     * @param pdfIS The input stream of the PDF file to create the RAG index from.
     * @param callback A callback interface to report progress and handle success or error during the RAG creation process.
     */
    public static void createRAG(InputStream pdfIS, RAGCallback callback) {

        PDFExtractor pdfExtractor;

        // Read the PDF file
        try {
            callback.onProgress(0.0f, RAGCallback.STEP_READING_PDF);
            pdfExtractor = new PDFExtractor(pdfIS);
            callback.onProgress(1.0f, RAGCallback.STEP_READING_PDF);
        } catch (Exception e) {
            callback.onError(e, RAGCallback.STEP_READING_PDF);
            return;
        }

        String[] chunks;

        // Chunk the PDF text
        try {
            callback.onProgress(0.0f, RAGCallback.STEP_CHUNKING_PDF);
            chunks = pdfExtractor.extractChunks();
            callback.onProgress(1.0f, RAGCallback.STEP_CHUNKING_PDF);
        } catch (Exception e) {
            callback.onError(e, RAGCallback.STEP_CHUNKING_PDF);
            return;
        }

        // Generate token and embedding for each chunk, and store them in ObjectBox
        try {
            callback.onProgress(0.0f, RAGCallback.STEP_CREATING_RAG);
            for (int i = 0; i < chunks.length; i++) {
                Tokenizer.Tokens tokens = tokenizer.tokenize(chunks[i]);
                float[] embedding = embeddingEngine.getEmbedding(tokens.inputIds, tokens.attentionMask, tokens.tokenTypeIds);
                objectBox.putItem(chunks[i], embedding);
                callback.onProgress((float) (i + 1) / chunks.length, RAGCallback.STEP_CREATING_RAG);
            }
            callback.onProgress(1.0f, RAGCallback.STEP_CREATING_RAG);
        } catch (Exception e) {
            callback.onError(e, RAGCallback.STEP_CREATING_RAG);
            return;
        }

        callback.onSuccess();
    }

    /**
     * Query the RAG index with the given query string and return the most relevant chunks based on cosine similarity.
     * @param query The query string to search for in the RAG index.
     * @return An array of the most relevant chunks from the RAG index based on cosine similarity to the query.
     */
    public static String[] queryRAG(String query) throws RuntimeException {

        // Tokenize the query and get its embedding
        Tokenizer.Tokens tokens = tokenizer.tokenize(query);
        try {
            float[] queryEmbedding = embeddingEngine.getEmbedding(tokens.inputIds, tokens.attentionMask, tokens.tokenTypeIds);

            // Query ObjectBox for the most similar chunks based on cosine similarity
            return objectBox.getItems(queryEmbedding, Config.N_RESULTS);
        }

        // Convert any OrtException thrown to avoid "cannot access OrtException"
        catch (OrtException e) { throw new RuntimeException(e); }
    }
}
