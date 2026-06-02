package com.example.anhilyx.resc_ai;

import io.objectbox.annotation.Entity;
import io.objectbox.annotation.Id;
import io.objectbox.annotation.HnswIndex;

@Entity
public class DocumentChunk {
    @Id
    public long id;

    public String text;

    @HnswIndex(dimensions = 384)
    public float[] embedding;

    public DocumentChunk() {}

    public DocumentChunk(String text, float[] embedding) {
        this.text = text;
        this.embedding = embedding;
    }
}
