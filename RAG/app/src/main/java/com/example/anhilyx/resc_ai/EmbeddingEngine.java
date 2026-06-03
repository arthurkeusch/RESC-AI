package com.example.anhilyx.resc_ai;

import android.content.Context;
import android.content.SharedPreferences;
import android.util.Log;

import androidx.annotation.NonNull;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import okhttp3.OkHttpClient;
import okhttp3.Request;

public class EmbeddingEngine {

    private static final int N_TOKENS = 512;

    private static final String TAG = "EmbeddingEngine";
    private final OkHttpClient client = new OkHttpClient();
    private final Context context;

    private OrtEnvironment env;
    private OrtSession session;
    private SimpleBertTokenizer tokenizer;

    public EmbeddingEngine(Context context) {
        this.context = context;
    }

    public interface ModelDownloadCallback {
        void onSuccess();
        void onError(String error);
    }

    /**
     * Create the callback for the HTTP request.
     * @param outFile The file to write to.
     * @param callback The callback to invoke.
     * @return The callback for the HTTP request.
     */
    private okhttp3.Callback getOkHttp3Callback(File outFile, ModelDownloadCallback callback) {
        return new okhttp3.Callback() {
            @Override
            public void onFailure(@NonNull okhttp3.Call call, @NonNull java.io.IOException e) {
                callback.onError(e.getMessage());
            }

            @Override
            public void onResponse(@NonNull okhttp3.Call call, @NonNull okhttp3.Response response) {

                // On unsuccessful requests, return an error
                if (!response.isSuccessful()) {
                    callback.onError("HTTP Error Code: " + response.code());
                    return;
                }

                // On successful requests, try to write the file to the disk
                try (InputStream is = response.body().byteStream();
                     FileOutputStream fos = new FileOutputStream(outFile)) {
                    byte[] buffer = new byte[8192];
                    int read;
                    while ((read = is.read(buffer)) != -1) {
                        fos.write(buffer, 0, read);
                    }
                    callback.onSuccess();
                } catch (Exception e) {
                    callback.onError(e.getMessage());
                }
            }
        };
    }

    /**
     * Download the required files for the given model.
     * @param model The model to download.
     * @param callback The callback to invoke.
     */
    public void downloadModel(ModelDescriptor model, ModelDownloadCallback callback) {

        // Prepare the local files
        File modelFile = new File(context.getFilesDir(), "model.onnx");
        File vocabFile = new File(context.getFilesDir(), "vocab.txt");
        if (modelFile.exists()) {
            callback.onSuccess();
            return;
        }

        // Download the model
        client
                .newCall(
                        new Request.Builder().url(model.getModelUrl()).build()
                )
                .enqueue(
                        getOkHttp3Callback(modelFile, callback)
                );

        // Download the vocab
        client
                .newCall(
                        new Request.Builder().url(model.getVocabUrl()).build()
                )
                .enqueue(
                        getOkHttp3Callback(vocabFile, callback)
                );

        // Save the model dimension
        SharedPreferences prefs = context.getSharedPreferences("rag_prefs", Context.MODE_PRIVATE);
        prefs.edit()
                .putInt("model_dimension", model.getDimension())
                .apply();
    }

    /**
     * Ensure the model is loaded.
     */
    private synchronized void loadModel() throws FileNotFoundException, OrtException {

        // The model is already loaded
        if (session != null) { return; }

        // Check for the existence of the model files
        File modelFile = new File(context.getFilesDir(), "model.onnx");
        File vocabFile = new File(context.getFilesDir(), "vocab.txt");
        if (!modelFile.exists()) {
            throw new FileNotFoundException("Model file not found");
        } else if (!vocabFile.exists()) {
            throw new FileNotFoundException("Vocab file not found");
        }

        // Initialize ONNX Runtime
        if (env == null) {
            env = OrtEnvironment.getEnvironment();
        }

        // Initialize ONNX Model
        session = env.createSession(modelFile.getAbsolutePath(), new OrtSession.SessionOptions());
        tokenizer = new SimpleBertTokenizer(vocabFile);
    }

    /**
     * Get the embedding for the given text.
     * @param text The text to embed.
     * @return The embedding.
     */
    public float[] generateEmbedding(String text) {

        // Retrieve the dimension
        SharedPreferences prefs = context.getSharedPreferences("rag_prefs", Context.MODE_PRIVATE);
        int dimension = prefs.getInt("model_dimension", 0);

        // Initialize empty embedding
        float[] embedding = new float[dimension];
        if (text == null || text.isEmpty()) return embedding;

        // Load the model if not already done
        try { loadModel(); }
        catch (Exception e) {
            Log.e(TAG, "Error while loading model: " + e.getMessage());
            return embedding;
        }

        try {
            // Tokenize the input text and define tensor dimensions
            TokenizerResult tokens = tokenizer.tokenize(text);
            long[] shape = new long[]{1, N_TOKENS};

            // Convert token arrays into native ONNX tensors
            OnnxTensor inputIdsTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(tokens.inputIds), shape);
            OnnxTensor attentionMaskTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(tokens.attentionMask), shape);
            OnnxTensor tokenTypeIdsTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(tokens.tokenTypeIds), shape);

            // Prepare the input map expected by the model
            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("input_ids", inputIdsTensor);
            inputs.put("attention_mask", attentionMaskTensor);
            inputs.put("token_type_ids", tokenTypeIdsTensor);

            // Execute the ONNX model inference
            try (OrtSession.Result results = session.run(inputs)) {
                // Extract the token embeddings from the output
                float[][][] outputEmbeddings = (float[][][]) results.get(0).getValue();

                // Aggregate token embeddings using mean pooling (ignoring padding)
                int validTokens = 0;
                for (int i = 0; i < N_TOKENS; i++) {
                    if (tokens.attentionMask[i] == 1L) {
                        validTokens++;
                        for (int j = 0; j < dimension; j++) {
                            embedding[j] += outputEmbeddings[0][i][j];
                        }
                    }
                }

                if (validTokens > 0) {
                    for (int j = 0; j < dimension; j++) {
                        embedding[j] /= validTokens;
                    }
                }

                // Apply L2 normalization to the final embedding vector
                float sum = 0;
                for (float v : embedding) sum += v * v;
                float norm = (float) Math.sqrt(sum);
                if (norm > 0) {
                    for (int i = 0; i < dimension; i++) {
                        embedding[i] /= norm;
                    }
                }
            }

            // Explicitly close native tensors to avoid memory leaks
            inputIdsTensor.close();
            attentionMaskTensor.close();
            tokenTypeIdsTensor.close();

        } catch (Exception e) {
            Log.e(TAG, "Error during embeddings inference: " + e.getMessage());
        }

        return embedding;
    }

    // Class to hold the tokenizer result
    private static class TokenizerResult {
        long[] inputIds;
        long[] attentionMask;
        long[] tokenTypeIds;

        TokenizerResult(long[] inputIds, long[] attentionMask, long[] tokenTypeIds) {
            this.inputIds = inputIds;
            this.attentionMask = attentionMask;
            this.tokenTypeIds = tokenTypeIds;
        }
    }

    // Class to hold the vocabulary
    private static class SimpleBertTokenizer {
        private final Map<String, Integer> vocab = new HashMap<>();

        SimpleBertTokenizer(File vocabFile) {
            // Read the vocab from the file
            try (BufferedReader reader = new BufferedReader(new FileReader(vocabFile))) {
                String line;
                int id = 0;
                while ((line = reader.readLine()) != null) {
                    vocab.put(line.trim(), id++);
                }
            } catch (Exception e) {
                Log.e(TAG, "Error while loading vocab file: " + e.getMessage());
            }
        }

        /**
         * Tokenize the given text.
         * This method should follow the "official BERT tokenizer algorithm"
         * @param text The text to tokenize.
         * @return The tokenized sequence.
         */
        TokenizerResult tokenize(String text) {

            // Initialize token list and add CLS token
            List<Long> ids = new ArrayList<>();
            ids.add(101L);

            // Clean text and split into individual words
            String[] words = text.toLowerCase().replaceAll("[^a-zA-Z0-9 ]", "").split("\\s+");

            // Process each word into tokens or subwords
            for (String word : words) {
                if (word.isEmpty()) continue;
                if (ids.size() >= N_TOKENS - 1) break;

                // Check if the full word exists in the vocabulary
                if (vocab.containsKey(word)) {
                    ids.add((long) vocab.get(word));
                    continue;
                }

                // Execute WordPiece algorithm to find subwords
                int start = 0;
                List<Long> subwordIds = new ArrayList<>();
                boolean isUnknown = false;

                while (start < word.length()) {
                    int end = word.length();
                    String curSubword = null;

                    while (start < end) {
                        String substr = word.substring(start, end);
                        if (start > 0) {
                            substr = "##" + substr;
                        }

                        if (vocab.containsKey(substr)) {
                            curSubword = substr;
                            break;
                        }
                        end--;
                    }

                    if (curSubword == null) {
                        isUnknown = true;
                        break;
                    }

                    subwordIds.add((long) vocab.get(curSubword));
                    start = end;
                }

                // Append subword tokens or fall back to UNK token
                if (isUnknown) {
                    if (ids.size() >= N_TOKENS - 1) break;
                    ids.add(100L);
                } else {
                    for (Long subId : subwordIds) {
                        if (ids.size() >= N_TOKENS - 1) break;
                        ids.add(subId);
                    }
                }
            }

            // Add SEP token to finalize the sequence
            ids.add(102L);

            // Initialize model input arrays
            long[] inputIds = new long[N_TOKENS];
            long[] attentionMask = new long[N_TOKENS];
            long[] tokenTypeIds = new long[N_TOKENS];

            // Apply padding and generate attention mask
            for (int i = 0; i < N_TOKENS; i++) {
                if (i < ids.size()) {
                    inputIds[i] = ids.get(i);
                    attentionMask[i] = 1L;
                } else {
                    inputIds[i] = 0L;
                    attentionMask[i] = 0L;
                }
                tokenTypeIds[i] = 0L;
            }

            return new TokenizerResult(inputIds, attentionMask, tokenTypeIds);
        }
    }
}