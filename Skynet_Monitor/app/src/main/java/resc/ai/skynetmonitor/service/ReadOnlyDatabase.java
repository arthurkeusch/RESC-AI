package resc.ai.skynetmonitor.service;

import android.content.Context;
import android.util.Log;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;

public class ReadOnlyDatabase {

    private static final int HEADER_SIZE = 10;  // 4 + 4 + 1 + 1 bytes

    private final String databaseName;
    private final RandomAccessFile inputStream;
    private final RandomAccessFile textsInputStream;

    private final int stringLength;
    private final int embeddingDimension;
    private final boolean isSorted;
    private final boolean longTexts;

    private final int lineSize;
    private final int linesCount;

    private final ByteBuffer stringBuffer;
    private final ByteBuffer embeddingBuffer;

    /**
     * Constructor for ReadOnlyDatabase.
     * @param context  The application context.
     * @param dbName   The name of the database file in assets.
     */
    public ReadOnlyDatabase(Context context, String dbName) {
        this.databaseName = dbName;

        // Copy the database from assets if needed
        File dbFile = new File(context.getFilesDir(), dbName);
        if (!dbFile.exists()) {
            try {
                InputStream assetStream = context.getAssets().open(dbName + ".rod");
                java.io.FileOutputStream outputStream = new java.io.FileOutputStream(dbFile);
                byte[] buffer = new byte[1024];
                int length;
                while ((length = assetStream.read(buffer)) > 0) {
                    outputStream.write(buffer, 0, length);
                }
                outputStream.close();
                assetStream.close();
            } catch (IOException e) {
                Log.d("ROD", "Error copying database '" + dbName + "': " + e.getMessage());
                throw new RuntimeException(e);
            }
        }

        // Initialize read-only database connection
        // The database file is located in the assets directory
        try {
            this.inputStream = new RandomAccessFile(dbFile, "r");
        } catch (IOException e) {
            Log.d("ROD", "Error opening database '" + dbName + "': " + e.getMessage());
            throw new RuntimeException(e);
        }

        Log.d("ROD", "Read-only database '" + dbName + "' initialized successfully.");

        // Initialize data (contained at the start of the database file)
        try {
            ByteBuffer buffer = ByteBuffer.allocate(HEADER_SIZE); // 4 + 4 + 1 bytes
            int read = inputStream.read(buffer.array());
            if (read != HEADER_SIZE) throw new IOException("Failed to read database metadata.");

            // 1st: 4 bytes: String length (int)
            this.stringLength = buffer.getInt(0);
            this.stringBuffer = ByteBuffer.allocate(this.stringLength);  // Already converted to UTF-32 size (4 bytes) if needed
            // 2nd: 4 bytes: Embedding dimension (int)
            this.embeddingDimension = buffer.getInt(4);
            this.embeddingBuffer = ByteBuffer.allocate(this.embeddingDimension * 4); // float: 4 bytes
            // 3rd: 1 byte: Is sorted? (boolean)
            this.isSorted = buffer.get(8) != 0;
            // 4th: 1 byte: Has long texts? (boolean)
            this.longTexts = buffer.get(9) != 0;

        } catch (IOException e) {
            Log.d("ROD", "Error reading metadata from database '" + dbName + "': " + e.getMessage());
            throw new RuntimeException(e);
        }

        // Load texts file if long texts
        if (longTexts) {
            File textsFile = new File(context.getFilesDir(), dbName + ".rot");
            if (!textsFile.exists()) {
                try {
                    InputStream assetStream = context.getAssets().open(dbName + ".rot");
                    java.io.FileOutputStream outputStream = new java.io.FileOutputStream(textsFile);
                    byte[] buffer = new byte[1024];
                    int length;
                    while ((length = assetStream.read(buffer)) > 0) {
                        outputStream.write(buffer, 0, length);
                    }
                    outputStream.close();
                    assetStream.close();
                } catch (IOException e) {
                    Log.d("ROD", "Error copying texts file for database '" + dbName + "': " + e.getMessage());
                    throw new RuntimeException(e);
                }
            }

            try {
                this.textsInputStream = new RandomAccessFile(textsFile, "r");
            } catch (IOException e) {
                Log.d("ROD", "Error opening texts file for database '" + dbName + "': " + e.getMessage());
                throw new RuntimeException(e);
            }
        } else this.textsInputStream = null;

        // Retrieve database size
        try {
            long totalSize = inputStream.length();
            this.lineSize = this.stringLength + (this.embeddingDimension * 4);
            this.linesCount = (int) ((totalSize - HEADER_SIZE) / this.lineSize); // Subtract metadata size
        } catch (IOException e) {
            Log.d("ROD", "Error calculating size of database '" + dbName + "': " + e.getMessage());
            throw new RuntimeException(e);
        }
    }

    /**
     * Get the number of lines in the database.
     * @return The number of lines.
     */
    public int size() {
        return linesCount;
    }

    /**
     * Get the index of a string in the database.
     * This method is more efficient if the database is sorted, but works for unsorted databases as well.
     * @param str The string to search for.
     * @return The index of the string, or -1 if not found.
     */
    public int getIndex(String str) {

        // Binary search for sorted database
        if (isSorted) {
            int left = 0;
            int right = linesCount - 1;
            while (left <= right) {
                int mid = left + (right - left) / 2;
                String currentString = getString(mid);  // Remove padding null characters
                int cmp = currentString.compareTo(str);
                if (cmp == 0) {
                    return mid;
                } else if (cmp < 0) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }

        // Linear search for unsorted database
        else {
            for (int i = 0; i < linesCount; i++) {
                String currentString = getString(i);  // Remove padding null characters
                if (currentString.equals(str)) {
                    return i;
                }
            }
        }

        return -1; // Not found
    }

    /**
     * Get the embeddings for a given index.
     * @param index The index of the string.
     * @return The embeddings as a float array.
     */
    public float[] getEmbeddings(int index) {

        try {
            // Calculate the offset to the embeddings
            long offset = HEADER_SIZE + (long) index * lineSize + (long) stringLength;
            inputStream.seek(offset);
            // Read the embeddings
            int read = inputStream.read(embeddingBuffer.array());
            if (read != embeddingDimension * 4) {
                throw new IOException("Failed to read embeddings for index " + index);
            }

            // Convert ByteBuffer to float array
            float[] embeddings = new float[embeddingDimension];
            for (int i = 0; i < embeddingDimension; i++) {
                embeddings[i] = embeddingBuffer.getFloat(i * 4);
            }
            return embeddings;

        } catch (IOException e) {
            Log.d("ROD", "Error reading embeddings from database '" + databaseName + "': " + e.getMessage());
            throw new RuntimeException(e);
        }
    }

    /**
     * Get the string at a given index.
     * @param index The index of the string.
     * @return The string.
     */
    public String getString(int index) {

        try {
            // Calculate the offset to the string
            long offset = HEADER_SIZE + (long) index * lineSize;
            inputStream.seek(offset);
            // Read the string
            int read = inputStream.read(stringBuffer.array());
            if (read != stringLength) {
                throw new IOException("Failed to read string for index " + index);
            }

            // Convert ByteBuffer to String (if not long texts)
            if (!longTexts) {
                StringBuilder sb = new StringBuilder();
                for (int i = 0; i < stringLength; i += 4) {
                    int codepoint = stringBuffer.getInt(i);
                    sb.appendCodePoint(codepoint);
                }
                return sb.toString().replace("\0", "");  // Remove padding null characters
            }

            // Retrieve the text (if long texts)
            else {
                long sOffset = stringBuffer.getLong(0);
                long sLength = stringBuffer.getLong(8);

                textsInputStream.seek(sOffset);
                ByteBuffer longTextBuffer = ByteBuffer.allocate((int) sLength);
                int longRead = textsInputStream.read(longTextBuffer.array());
                if (longRead != sLength) {
                    throw new IOException("Failed to read long text for index " + index);
                }
                return new String(longTextBuffer.array(), StandardCharsets.UTF_8);
            }

        } catch (IOException e) {
            Log.d("ROD", "Error reading string from database '" + databaseName + "': " + e.getMessage());
            throw new RuntimeException(e);
        }
    }
}
