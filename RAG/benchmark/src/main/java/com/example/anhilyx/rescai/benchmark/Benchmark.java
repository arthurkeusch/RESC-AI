package com.example.anhilyx.rescai.benchmark;

import android.content.Context;
import android.util.Log;

import com.example.anhilyx.rescai.rag.Config;
import com.example.anhilyx.rescai.rag.RAG;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayInputStream;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicLong;

/**
 * The Benchmark class is responsible for running benchmarks on different HuggingFace repositories using the RAG implementation.
 * It defines the structure of the benchmark, including models, documents, and prompts, and manages the execution flow of the benchmark while reporting progress and results.
 */
public class Benchmark {

    /**
     * A data class that represents a single model for the benchmark, which contains multiple documents.
     */
    public static class ModelBenchmark {

        public final String repository;
        public final DocumentBenchmark[] documents;

        /**
         * Constructor for the ModelBenchmark class.
         * @param repository The name of the HuggingFace repository to be used for the benchmark.
         * @param documents An array of DocumentBenchmark objects that are associated with the model.
         */
        public ModelBenchmark(String repository, DocumentBenchmark[] documents) {
            this.repository = repository;
            this.documents = documents;
        }
    }

    /**
     * A data class that represents a single document for the benchmark, which contains multiple prompts.
     */
    public static class DocumentBenchmark {

        public final String name;
        public final byte[] data;
        public final PromptBenchmark[] prompts;

        /**
         * Constructor for the DocumentBenchmark class.
         * @param name The name of the document to be used for the benchmark.
         * @param data The byte array of the document data to be used for the benchmark.
         * @param prompts An array of PromptBenchmark objects that are associated with the document.
         */
        public DocumentBenchmark(String name, byte[] data, PromptBenchmark[] prompts) {
            this.name = name;
            this.data = data;
            this.prompts = prompts;
        }
    }

    /**
     * A data class that represents a single prompt for the benchmark.
     */
    public static class PromptBenchmark {

        public final String prompt;

        /**
         * Constructor for the PromptBenchmark class.
         * @param prompt The prompt to be used for the benchmark.
         */
        public PromptBenchmark(String prompt) {
            this.prompt = prompt;
        }
    }

    /**
     * A data class that represents the progress of the benchmark.
     */
    public static class BenchmarkProgress {

        public float modelsProgress = 0.0f;
        public String currentModel = "";

        public float documentsProgress = 0.0f;
        public String currentDocument = "";

        public float promptsProgress = 0.0f;
        public String currentPrompt = "";

        /**
         * Constructor for the BenchmarkProgress class.
         * @param state The current state of the benchmark, which contains information about the current model, document, and prompt being processed.
         */
        public BenchmarkProgress(BenchmarkState state) {

            if (state.modelIdx >= 0) {
                this.modelsProgress = (float) state.modelIdx / state.models.length;
                ModelBenchmark model = state.models[Math.max(0, state.modelIdx)];
                this.currentModel = model.repository;

                if (state.documentIdx >= 0) {
                    this.documentsProgress = (float) state.documentIdx / model.documents.length;
                    DocumentBenchmark document = model.documents[Math.max(0, state.documentIdx)];
                    this.currentDocument = document.name;

                    if (state.promptIdx >= 0) {
                        this.promptsProgress = (float) state.promptIdx / document.prompts.length;
                        PromptBenchmark prompt = document.prompts[state.promptIdx];
                        this.currentPrompt = prompt.prompt;
                    }
                }
            }
        }
    }

    /**
     * An interface for a callback to report progress during the benchmark.
     */
    public interface BenchmarkProgressCallback {

        /**
         * Called to report progress during the benchmark.
         * @param progress A float value between 0 and 1 representing the overall progress of the benchmark.
         * @param step An integer representing the current step of the benchmark (e.g., model, document, prompt).
         * @param benchmarkProgress An object containing detailed progress information for the current model, document, and prompt.
         */
        void onProgress(float progress, int step, BenchmarkProgress benchmarkProgress);
    }

    /**
     * A data class that represents the state of the benchmark.
     */
    public static class BenchmarkState {

        public final ModelBenchmark[] models;
        public final Context context;
        public final BenchmarkProgressCallback callback;
        public final CompletableFuture<JSONObject> future;
        public int modelIdx = -1;  // Start at -1 because all increments happen at the start of the loop
        public int documentIdx = -1;  // Start at -1 because all increments happen at the start of the loop
        public int promptIdx = -1;  // Start at -1 because all increments happen at the start of the loop
        public final JSONObject results = new JSONObject();

        /**
         * Constructor for the BenchmarkState class.
         * @param models An array of ModelBenchmark objects that represents which combinations of models, documents and prompts will be used.
         * @param context The Android context, used to access the app's file directory for loading the tokenizer and model files.
         * @param callback A callback interface to report progress during the benchmark.
         * @param future A CompletableFuture that will be completed when the benchmark is finished, containing the results of the benchmark as a JSONObject.
         */
        public BenchmarkState(ModelBenchmark[] models, Context context, BenchmarkProgressCallback callback, CompletableFuture<JSONObject> future) {
            this.models = models;
            this.context = context;
            this.callback = callback;
            this.future = future;
        }
    }

    private final ModelBenchmark[] models;

    /**
     * Constructor for the Benchmark class.
     * @param models An array of ModelBenchmark objects that represents which combinations of models, documents and prompts will be used.
     */
    public Benchmark(ModelBenchmark[] models) {
        this.models = models;
    }

    /**
     * Runs the benchmark with the specified context and progress callback.
     * @param context The Android context, used to access the app's file directory for loading the tokenizer and model files.
     * @param callback A callback interface to report progress during the benchmark.
     * @return A CompletableFuture that will be completed when the benchmark is finished, containing the results of the benchmark as a JSONObject.
     */
    public CompletableFuture<JSONObject> run(Context context, BenchmarkProgressCallback callback) {

        // Initialize the benchmark state
        CompletableFuture<JSONObject> future = new CompletableFuture<>();
        BenchmarkState state = new BenchmarkState(models, context, callback, future);

        // Run the benchmark
        state.modelIdx = -1;  // Reset model index to be safe
        benchmarkModel(state);
        return future;
    }

    /**
     * Runs the benchmark for the current model in the benchmark state, and recursively calls itself to move on to the next model until all models have been processed.
     * @param state The current state of the benchmark, which contains information about the current model, document, and prompt being processed, as well as the results collected so far and the callback to report progress.
     */
    private void benchmarkModel(BenchmarkState state) {

        // Increment the model index and check if we have finished all models
        state.modelIdx++;
        if (state.modelIdx >= models.length) {
            state.future.complete(state.results);
            return;
        }

        // Run the benchmark for the current model
        ModelBenchmark model = models[state.modelIdx];
        AtomicLong startTime = new AtomicLong();
        startTime.set(System.currentTimeMillis());
        RAG.init(state.context, new RAG.RAGCallback() {
            @Override
            public void onSuccess() {

                // Stop the timer for the current model
                long endTime = System.currentTimeMillis();
                long elapsedTime = endTime - startTime.get();

                // On success, record the time taken for the current model and prepare the documents' results object
                try {
                    state.results
                            .put(
                                    model.repository,
                                    new JSONObject()
                                            .put("success", true)
                                            .put("time_ms", elapsedTime)
                                            .put("documents", new JSONObject())
                            );
                } catch (JSONException e) {
                    Log.e("BENCHMARK", "", e);
                }

                // Move on to the first document for the current model
                benchmarkDocument(state);
            }

            @Override
            public void onError(Exception error, int step) {
                Log.w("BENCHMARK", "", error);

                // On error, record the failure for the current model
                try {
                    state.results
                            .put(
                                model.repository,
                                new JSONObject()
                                        .put("success", false)
                                        .put("error", error.getMessage())
                            );
                } catch (JSONException e) {
                    Log.e("BENCHMARK", "", error);
                    Log.e("BENCHMARK", "", e);
                }

                // Move on to the next model even if there was an error
                benchmarkModel(state);
            }

            @Override
            public void onProgress(float progress, int step) {
                state.callback.onProgress(
                        progress,
                        step,
                        new BenchmarkProgress(state)
                );
            }
        }, model.repository);
    }

    /**
     * Runs the benchmark for the current document in the benchmark state, and recursively calls itself to move on to the next document until all documents for the current model have been processed. Once all documents are processed, it moves on to the next model.
     * @param state The current state of the benchmark, which contains information about the current model, document, and prompt being processed, as well as the results collected so far and the callback to report progress.
     */
    private void benchmarkDocument(BenchmarkState state) {

            // Increment the document index and check if we have finished all documents for the current model
            state.documentIdx++;
            ModelBenchmark model = models[state.modelIdx];
            if (state.documentIdx >= model.documents.length) {
                state.documentIdx = -1;  // Reset document index for the next model
                benchmarkModel(state);
                return;
            }

            // Run the benchmark for the current document
            DocumentBenchmark document = model.documents[state.documentIdx];
            AtomicLong startTime = new AtomicLong();
            startTime.set(System.currentTimeMillis());
            RAG.emptyRAG();
            RAG.inflateRAG(new ByteArrayInputStream(document.data), new RAG.RAGCallback() {
                @Override
                public void onSuccess() {

                    // Stop the timer for the current document
                    long endTime = System.currentTimeMillis();
                    long elapsedTime = endTime - startTime.get();

                    // On success, record the time taken for the current document and prepare the prompts' results object
                    try {
                        state.results
                                .getJSONObject(model.repository)
                                .getJSONObject("documents")
                                .put(
                                        document.name,
                                        new JSONObject()
                                                .put("success", true)
                                                .put("time_ms", elapsedTime)
                                                .put("prompts", new JSONObject())
                                );
                    } catch (JSONException e) {
                        Log.e("BENCHMARK", "", e);
                    }

                    // Move on to the first prompt for the current document
                    benchmarkPrompt(state);
                }

                @Override
                public void onError(Exception error, int step) {
                    Log.w("BENCHMARK", "", error);

                    // On error, record the failure for the current document
                    try {
                        state.results
                                .getJSONObject(model.repository)
                                .getJSONObject("documents")
                                .put(
                                        document.name,
                                        new JSONObject()
                                                .put("success", false)
                                                .put("error", error.getMessage())
                                );
                    } catch (JSONException e) {
                        Log.e("BENCHMARK", "", error);
                        Log.e("BENCHMARK", "", e);
                    }

                    // Move on to the next document even if there was an error
                    benchmarkDocument(state);
                }

                @Override
                public void onProgress(float progress, int step) {
                    state.callback.onProgress(
                            progress,
                            step,
                            new BenchmarkProgress(state)
                    );
                }
            });
    }

    /**
     * Runs the benchmark for the current prompt in the benchmark state, and recursively calls itself to move on to the next prompt until all prompts for the current document have been processed. Once all prompts are processed, it moves on to the next document.
     * @param state The current state of the benchmark, which contains information about the current model, document, and prompt being processed, as well as the results collected so far and the callback to report progress.
     */
    private void benchmarkPrompt(BenchmarkState state) {

        // Loop through the prompts for the current document
        ModelBenchmark model = models[state.modelIdx];
        DocumentBenchmark document = model.documents[state.documentIdx];
        while (++state.promptIdx < document.prompts.length) {

            // Notify progress for the current prompt
            state.callback.onProgress(
                    0.0f,
                    RAG.RAGCallback.STEP_QUERYING_RAG,
                    new BenchmarkProgress(state)
            );

            // Run the benchmark for the current prompt
            PromptBenchmark prompt = document.prompts[state.promptIdx];
            AtomicLong startTime = new AtomicLong();
            startTime.set(System.currentTimeMillis());
            String[] retrievedChunks = RAG.queryRAG(prompt.prompt);

            // Stop the timer for the current prompt
            long endTime = System.currentTimeMillis();
            long elapsedTime = endTime - startTime.get();

            // On success, record the time taken and the retrieved chunks for the current prompt
            try {
                state.results
                        .getJSONObject(model.repository)
                        .getJSONObject("documents")
                        .getJSONObject(document.name)
                        .getJSONObject("prompts")
                        .put(
                                prompt.prompt,
                                new JSONObject()
                                        .put("success", true)
                                        .put("time_ms", elapsedTime)
                                        .put("retrieved_chunks", new JSONArray(retrievedChunks))
                        );
            } catch (JSONException e) {
                Log.e("BENCHMARK", "", e);
            }

            // Notify progress for the current prompt
            state.callback.onProgress(
                    1.0f,
                    RAG.RAGCallback.STEP_QUERYING_RAG,
                    new BenchmarkProgress(state)
            );
        }

        // After finishing all prompts for the current document, move on to the next document
        state.promptIdx = -1;  // Reset prompt index for the next document
        benchmarkDocument(state);
    }
}
