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
        int length = Math.min(rawIds.length, tokenizer.getMaxLength());
        long[] inputIds = new long[length];
        long[] attentionMask = new long[length];
        long[] tokenTypeIds = new long[length];
        System.arraycopy(rawIds, 0, inputIds, 0, length);
        System.arraycopy(rawAttentionMask, 0, attentionMask, 0, length);
        System.arraycopy(rawTokenTypeIds, 0, tokenTypeIds, 0, length);
        inputIds[length - 1] = 102L;  // We always set the last token to the EOS token (102), for cases where the input is truncated

        return new Tokens(inputIds, attentionMask, tokenTypeIds);
    }
}
