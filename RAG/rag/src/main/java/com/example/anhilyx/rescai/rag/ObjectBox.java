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

    protected Box<Item> itemBox;

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
            Box<Item> itemBox = boxStore.boxFor(Item.class);
            itemBox.removeAll();
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

        itemBox = boxStore.boxFor(Item.class);
    }

    /**
     * Put an item into the ObjectBox database.
     * @param chunk the text chunk to store
     * @param embedding the embedding vector associated with the chunk
     */
    public void putItem(String chunk, float[] embedding) {

        Item item = new Item(chunk, embedding);
        itemBox.put(item);
    }

    /**
     * Get the nearest neighbor items from ObjectBox based on the target embedding vector.
     * @param target the target embedding vector to search for
     * @param n the number of nearest neighbors to retrieve
     * @return an array of text chunks corresponding to the nearest neighbors
     */
    public String[] getItems(float[] target, int n) {

        // Extract the nearest neighbors using ObjectBox's HNSW index
        Query<Item> query = itemBox.query()
                .nearestNeighbors(Item_.embedding, target, n)
                .build();
        List<Item> results = query.find();
        query.close();

        // Extract the texts from the results
        String[] chunks = new String[results.size()];
        for (int i = 0; i < results.size(); i++) {
            chunks[i] = results.get(i).chunk;
        }
        return chunks;
    }
}
