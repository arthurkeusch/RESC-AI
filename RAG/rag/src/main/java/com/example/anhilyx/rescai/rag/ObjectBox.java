package com.example.anhilyx.rescai.rag;

import android.content.Context;

import java.util.List;

import io.objectbox.Box;
import io.objectbox.BoxStore;
import io.objectbox.annotation.Entity;
import io.objectbox.annotation.HnswIndex;
import io.objectbox.annotation.Id;
import io.objectbox.annotation.VectorDistanceType;
import io.objectbox.query.Query;

/**
 * Utility class for managing ObjectBox database operations.
 */
public class ObjectBox {

    protected static BoxStore boxStore;

    protected Box<Item384> itemBox384;
    protected Box<Item768> itemBox768;
    protected Box<Item1024> itemBox1024;

    /**
     * Initialize ObjectBox. Should be called once at application startup.
     * @param context the application context
     * @return the initialized BoxStore instance
     */
    public static BoxStore init(Context context) {
        if (boxStore == null) {
            boxStore = MyObjectBox.builder()
                    .androidContext(context)
                    .build();
        }
        return boxStore;
    }

    /**
     * Empty the ObjectBox database by removing all items from the item box.
     */
    public static void empty() {
        if (boxStore != null) {
            boxStore.boxFor(Item384.class).removeAll();
            boxStore.boxFor(Item768.class).removeAll();
            boxStore.boxFor(Item1024.class).removeAll();
        }
    }

    /**
     * Constructor for ObjectBox.
     */
    public ObjectBox() {

        // Ensure that ObjectBox is initialized before trying to access the box
        if (boxStore == null) {
            throw new IllegalStateException("ObjectBox is not initialized. Call `ObjectBox.init(context)` first.");
        }

        itemBox384 = boxStore.boxFor(Item384.class);
        itemBox768 = boxStore.boxFor(Item768.class);
        itemBox1024 = boxStore.boxFor(Item1024.class);
    }

    /**
     * Put an item into the ObjectBox database.
     * @param chunk the text chunk to store
     * @param embedding the embedding vector associated with the chunk
     */
    public void putItem(String chunk, float[] embedding) {

        if (embedding.length == 384) {
            Item384 item = new Item384(chunk, embedding);
            itemBox384.put(item);
        } else if (embedding.length == 768) {
            Item768 item = new Item768(chunk, embedding);
            itemBox768.put(item);
        } else if (embedding.length == 1024) {
            Item1024 item = new Item1024(chunk, embedding);
            itemBox1024.put(item);
        } else {
            throw new IllegalArgumentException("Unsupported embedding dimension: " + embedding.length);
        }
    }

    /**
     * Get the nearest neighbor items from ObjectBox based on the target embedding vector.
     * @param target the target embedding vector to search for
     * @param n the number of nearest neighbors to retrieve
     * @return an array of text chunks corresponding to the nearest neighbors
     */
    public String[] getItems(float[] target, int n) {

        // Extract the nearest neighbors using ObjectBox's HNSW index
        if (target.length == 384) {
            Query<Item384> query = itemBox384.query()
                    .nearestNeighbors(Item384_.embedding, target, n)
                    .build();
            List<Item384> results = query.find();
            query.close();

            String[] chunks = new String[results.size()];
            for (int i = 0; i < results.size(); i++) {
                chunks[i] = results.get(i).chunk;
            }
            return chunks;

        } else if (target.length == 768) {
            Query<Item768> query = itemBox768.query()
                    .nearestNeighbors(Item768_.embedding, target, n)
                    .build();
            List<Item768> results = query.find();
            query.close();

            String[] chunks = new String[results.size()];
            for (int i = 0; i < results.size(); i++) {
                chunks[i] = results.get(i).chunk;
            }
            return chunks;

        } else if (target.length == 1024) {
            Query<Item1024> query = itemBox1024.query()
                    .nearestNeighbors(Item1024_.embedding, target, n)
                    .build();
            List<Item1024> results = query.find();
            query.close();

            String[] chunks = new String[results.size()];
            for (int i = 0; i < results.size(); i++) {
                chunks[i] = results.get(i).chunk;
            }
            return chunks;

        } else {
            throw new IllegalArgumentException("Unsupported embedding dimension: " + target.length);
        }

        // Extract the texts from the results
    }
}
