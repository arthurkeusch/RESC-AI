package com.example.anhilyx.rescai.rag;

import io.objectbox.annotation.Entity;
import io.objectbox.annotation.HnswIndex;
import io.objectbox.annotation.Id;
import io.objectbox.annotation.VectorDistanceType;

/**
 * Entity class representing an item in the ObjectBox database.
 * This class uses the 768-dimensional embedding vector for indexing and searching.
 */
@Entity
public class Item768 {

    @Id
    public long id;

    public String chunk;

    @HnswIndex(dimensions = 768, distanceType = VectorDistanceType.COSINE)
    public float[] embedding;

    public Item768() {}

    public Item768(String chunk, float[] embedding) {
        this.chunk = chunk;
        this.embedding = embedding;
    }
}
