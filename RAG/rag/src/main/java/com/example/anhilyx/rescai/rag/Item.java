package com.example.anhilyx.rescai.rag;


import io.objectbox.annotation.Entity;
import io.objectbox.annotation.HnswIndex;
import io.objectbox.annotation.Id;
import io.objectbox.annotation.VectorDistanceType;

/**
 * Entity class representing an item in the ObjectBox database.
 * Each item has a unique ID, a text chunk, and an embedding vector.
 */
@Entity
public class Item {
    @Id
    public long id;

    public String chunk;

    @HnswIndex(dimensions = Config.EMBEDDING_DIM, distanceType = VectorDistanceType.EUCLIDEAN)
    public float[] embedding;

    public Item() {}

    public Item(String chunk, float[] embedding) {
        this.chunk = chunk;
        this.embedding = embedding;
    }
}
