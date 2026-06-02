package com.example.anhilyx.resc_ai;

import io.objectbox.Box;
import io.objectbox.query.Query;
import io.objectbox.query.ObjectWithScore;
import java.util.ArrayList;
import java.util.List;

public class RagManager {

    private final Box<DocumentChunk> chunkBox;

    /**
     * RagManager constructor.
     */
    public RagManager() {
        this.chunkBox = App.getBoxStore().boxFor(DocumentChunk.class);
    }

    /**
     * Ingest a document chunk into the RAG.
     * @param text The text of the chunk.
     * @param embedding The embedding of the chunk.
     */
    public void ingestDocument(String text, float[] embedding) {
        DocumentChunk chunk = new DocumentChunk(text, embedding);
        chunkBox.put(chunk);
    }

    /**
     * Retrieve the most relevant chunks from the RAG.
     * @param queryEmbedding The embedding of the query.
     * @param maxResults The maximum number of results to retrieve.
     * @return The list of retrieved chunks.
     */
    public List<String> retrieveChunks(float[] queryEmbedding, int maxResults) {
        Query<DocumentChunk> query = chunkBox.query()
                .nearestNeighbors(DocumentChunk_.embedding, queryEmbedding, maxResults)
                .build();

        List<ObjectWithScore<DocumentChunk>> results = query.findWithScores();
        List<String> textChunks = new ArrayList<>();

        for (ObjectWithScore<DocumentChunk> result : results) {
            textChunks.add(result.get().text);
        }

        query.close();
        return textChunks;
    }
}