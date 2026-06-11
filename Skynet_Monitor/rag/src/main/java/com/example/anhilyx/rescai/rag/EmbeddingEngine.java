package com.example.anhilyx.rescai.rag;

import androidx.annotation.NonNull;

import java.io.File;
import java.nio.LongBuffer;
import java.util.HashMap;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

/**
 * Utility class for generating embeddings from input text using an ONNX model.
 */
public class EmbeddingEngine {

    protected OrtEnvironment environment;
    protected OrtSession session;

    /**
     * Constructor for the Embedder class.
     * @param modelFile The file containing the ONNX model to be used for generating embeddings.
     * @throws OrtException If there is an error initializing the ONNX Runtime environment or loading the model file.
     */
    public EmbeddingEngine(@NonNull File modelFile) throws OrtException {

        environment = OrtEnvironment.getEnvironment();
        session = environment.createSession(modelFile.getAbsolutePath(), new OrtSession.SessionOptions());
    }

    /**
     * Generate an embedding vector from the input token IDs, attention mask, and token type IDs.
     * @param inputIds The token IDs representing the input text.
     * @param attentionMask The attention mask indicating which tokens should be attended to.
     * @param tokenTypeIds The token type IDs indicating the segment of each token.
     * @return A float array representing the embedding vector for the input text.
     * @throws OrtException If there is an error during the ONNX model inference process.
     */
    public float[] getEmbedding(long[] inputIds, long[] attentionMask, long[] tokenTypeIds) throws OrtException {

        // Initialize empty embedding
        float[] embedding = new float[Config.EMBEDDING_DIM];
        long[] shape = new long[]{1, Config.N_TOKENS};

        // Prepare input tensors for the ONNX model
        try (
                OnnxTensor inputIdsTensor = OnnxTensor.createTensor(environment, LongBuffer.wrap(inputIds), shape);
                OnnxTensor attentionMaskTensor = OnnxTensor.createTensor(environment, LongBuffer.wrap(attentionMask), shape);
                OnnxTensor tokenTypeIdsTensor = OnnxTensor.createTensor(environment, LongBuffer.wrap(tokenTypeIds), shape)
        ) {
            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("input_ids", inputIdsTensor);
            inputs.put("attention_mask", attentionMaskTensor);
            inputs.put("token_type_ids", tokenTypeIdsTensor);

            // Execute the ONNX model inference
            try (OrtSession.Result results = session.run(inputs)) {
                float[][][] outputEmbeddings = (float[][][]) results.get(0).getValue();
                int validTokens = 0;

                // Aggregate token embeddings using mean pooling
                for (int i = 0; i < Config.N_TOKENS; i++) {
                    if (attentionMask[i] == 1L) {
                        validTokens++;
                        for (int j = 0; j < Config.EMBEDDING_DIM; j++) {
                            embedding[j] += outputEmbeddings[0][i][j];
                        }
                    }
                }

                // Average the token embeddings to get a single embedding for the entire input
                if (validTokens > 0) {
                    for (int j = 0; j < Config.EMBEDDING_DIM; j++) {
                        embedding[j] /= validTokens;
                    }
                }

                // Apply L2 normalization to the final embedding vector
                float sum = 0;
                for (float v : embedding) sum += v * v;
                float norm = (float) Math.sqrt(sum);
                if (norm > 0) {
                    for (int i = 0; i < Config.EMBEDDING_DIM; i++) {
                        embedding[i] /= norm;
                    }
                }
            }
        }

        return embedding;
    }
}
