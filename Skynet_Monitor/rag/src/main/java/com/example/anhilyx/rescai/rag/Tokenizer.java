package com.example.anhilyx.rescai.rag;

import androidx.annotation.NonNull;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;

/**
 * Utility class for tokenizing input text using a Hugging Face tokenizer.
 */
public class Tokenizer {

    private final HuggingFaceTokenizer tokenizer;

    /**
     * Result of tokenization, containing input IDs, attention mask, and token type IDs.
     */
    public static class Tokens {
        public final long[] inputIds;
        public final long[] attentionMask;
        public final long[] tokenTypeIds;

        /**
         * Constructor for TokenizerResult.
         * @param inputIds The token IDs representing the input text.
         * @param attentionMask The attention mask indicating which tokens should be attended to.
         * @param tokenTypeIds The token type IDs indicating the segment of each token.
         */
        private Tokens(long[] inputIds, long[] attentionMask, long[] tokenTypeIds) {
            this.inputIds = inputIds;
            this.attentionMask = attentionMask;
            this.tokenTypeIds = tokenTypeIds;
        }
    }

    /**
     * Constructor for the Tokenizer class.
     * @param tokenizerFile The file containing the tokenizer configuration (usually, `tokenizer.json`).
     * @throws IOException If there is an error loading the tokenizer file.
     */
    Tokenizer(@NonNull File tokenizerFile) throws IOException {

        Path path = tokenizerFile.toPath();
        tokenizer = HuggingFaceTokenizer.newInstance(path, null);

        // Throw an exception if the tokenizer's max length does not match the expected value
        if (tokenizer.getMaxLength() != Config.N_TOKENS) {
            throw new IllegalArgumentException("Tokenizer max length does not match expected value: " + tokenizer.getMaxLength());
        }
    }

    /**
     * Tokenize the input text and return the token IDs, attention mask, and token type IDs.
     * @param text The input text to tokenize.
     * @return A TokenizerResult containing the token IDs, attention mask, and token type IDs.
     */
    Tokens tokenize(@NonNull String text) {

        // Tokenize input text
        Encoding encoding = tokenizer.encode(text);
        long[] rawIds = encoding.getIds();
        long[] rawAttentionMask = encoding.getAttentionMask();
        long[] rawTokenTypeIds = encoding.getTypeIds();

        // Truncate the tokenized output to the model's maximum input size
        long[] inputIds = new long[Config.N_TOKENS];
        long[] attentionMask = new long[Config.N_TOKENS];
        long[] tokenTypeIds = new long[Config.N_TOKENS];
        int lengthToCopy = Math.min(rawIds.length, Config.N_TOKENS);
        System.arraycopy(rawIds, 0, inputIds, 0, lengthToCopy);
        System.arraycopy(rawAttentionMask, 0, attentionMask, 0, lengthToCopy);
        System.arraycopy(rawTokenTypeIds, 0, tokenTypeIds, 0, lengthToCopy);
        if (rawIds.length > Config.N_TOKENS) {
            inputIds[Config.N_TOKENS - 1] = 102L;
        }

        return new Tokens(inputIds, attentionMask, tokenTypeIds);
    }
}
