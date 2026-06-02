# ObjectBox & ONNX RAG Integration Technical Documentation

This document serves as a comprehensive technical guide for understanding and resuming work on the Retrieval-Augmented Generation (RAG) implementation within this Android project. It focuses heavily on the RAG pipeline: text extraction, embedding generation, vector database indexing, and query pipelines.

## 1. Setup & Dependencies

To enable on-device RAG, the project relies on ObjectBox (vector database), ONNX Runtime (on-device inference), and PDFBox (document parsing). To use these plugins & libraries, ensure the following configurations are present in your project files:

### `libs.versions.toml`
```toml
[versions]
# ... (existing versions)
objectbox = "5.4.2"
objectboxProcessor = "5.4.2"
onnxRuntime = "1.15.1"
pdfbox = "2.0.27.0"
okHttp = "5.3.2"

[libraries]
# ... (existing libraries)
objectbox-processor = { module = "io.objectbox:objectbox-processor", version.ref = "objectboxProcessor" }
onnx-runtime-android = { group = "com.microsoft.onnxruntime", name = "onnxruntime-android", version.ref = "onnxRuntime" }
pdfbox-android = { group = "com.tom-roush", name = "pdfbox-android", version.ref = "pdfbox" }
okHttp = { group = "com.squareup.okhttp3", name = "okhttp", version.ref = "okHttp" }

[plugins]
# ... (existing plugins)
objectbox = { id = "io.objectbox", version.ref = "objectbox" }
```

### `build.gradle.kts (Project)`
```kotlin
plugins {
    // ... (existing plugins)
    alias(libs.plugins.objectbox) apply false
}
```

### `build.gradle.kts (Module)`
```kotlin
plugins {
    // ... (existing plugins)
    alias(libs.plugins.objectbox)
}

// ... (existing configurations)

dependencies {
    // ... (existing dependencies)
    annotationProcessor(libs.objectbox.processor)
    implementation(libs.onnx.runtime.android)
    implementation(libs.pdfbox.android)
    implementation(libs.okHttp)
}
```

### `AndroidManifest.xml`
```xml
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:tools="http://schemas.android.com/tools">

    <uses-permission android:name="android.permission.INTERNET" />  </manifest>
```

---

## 2. Core Architecture Overview

The RAG pipeline operates entirely on-device, ensuring privacy and offline capabilities. It consists of two primary flows:
1. **Creation/Ingestion Flow:** PDF Document -> Text Extraction -> Text Chunking -> ONNX Embedding Generation -> ObjectBox Vector Store.
2. **Extraction/Querying Flow:** User Text Query -> ONNX Embedding Generation -> ObjectBox Nearest Neighbor Search -> Result Display.

---

## 3. RAG Pipeline Components

### 3.1. Vector Database Schema (`DocumentChunk.java` & `App.java`)
ObjectBox acts as the vector database for storing document chunks and their mathematical representations (embeddings).
- **`App.java`**: A custom Application class responsible for initializing the ObjectBox `BoxStore` singleton (`MyObjectBox.builder().androidContext(this).build()`).
- **`DocumentChunk.java`**: The core data model annotated with ObjectBox's `@Entity`.
  - Stores an `id` and the plain `text` of the chunk.
  - Contains a `float[] embedding` array annotated with `@HnswIndex(dimensions = 384)`. This annotation instructs ObjectBox to build a Hierarchical Navigable Small World (HNSW) graph for this property, enabling highly efficient Approximate Nearest Neighbor (ANN) searches in vector space. The `384` dimension precisely matches models like `all-MiniLM-L6-v2`.

### 3.2. Embedding Generation (`EmbeddingEngine.java`)
This class manages the lifecycle of the ONNX machine learning model used to convert text into vector embeddings.
- **Model Acquisition**: Uses `OkHttpClient` to asynchronously download the `.onnx` model files and `vocab.txt` definitions to local storage based on a selected `ModelDescriptor`.
- **Initialization**: Sets up the `OrtEnvironment` and `OrtSession` required to execute the ONNX graph on-device.
- **Tokenization**: Implements a custom `SimpleBertTokenizer` that maps text strings to vocabulary IDs. It automatically handles unknown tokens (falling back to `100L`), appends separator tokens (`102L`), and applies zero-padding to match model sequence length requirements.
- **Inference**: Constructs `inputIds`, `attentionMask`, and `tokenTypeIds` arrays, loads them into `OnnxTensor` objects, and runs the ONNX session to yield the final 384-dimensional `float[]` embeddings.

### 3.3. Database Wrapper (`RagManager.java`)
A repository class wrapping the ObjectBox `Box<DocumentChunk>` to abstract database transactions from the application logic.
- **Ingestion (`ingestDocument`):** Accepts a plain string and its computed `float[] embedding`, instantiates a new `DocumentChunk`, and persists it into the ObjectBox database.
- **Retrieval (`retrieveChunks`):** Constructs a vector search query leveraging the ObjectBox `QueryBuilder.nearestNeighbors(DocumentChunk_.embedding, queryEmbedding, maxResults)` method. It extracts the resulting `ObjectWithScore<DocumentChunk>` list and maps it back into plain string chunks for the UI.

---

## 4. Workflows & Configuration

### 4.1. Creation Workflow (`RagCreationFragment.java`)
Handles the document ingestion phase.
- **PDF Processing**: Leverages `PDFBoxResourceLoader` and `PDFTextStripper` to extract a continuous stream of pure text from user-selected PDF URIs.
- **Chunking Strategy**: Configured by the constant `CHUNK_SIZE = 1024`. To preserve semantic meaning and avoid slicing words in half, the algorithm establishes 1024 characters as a minimum chunk size. If the 1024th character lands inside a word, it increments character-by-character (`Character.isLetterOrDigit`) until it reaches whitespace or punctuation.
- **Execution**: Employs a single-thread `ExecutorService` to push chunks sequentially through the `EmbeddingEngine` and save them via the `RagManager`, ensuring the main UI thread remains responsive.

### 4.2. Extraction Workflow (`RagExtractionFragment.java`)
Handles the semantic search phase.
- **Query Parameter**: Configured by the constant `N_RESULTS = 5`, meaning the engine will strictly retrieve the top 5 most semantically relevant chunks for any given query.
- **Execution**: Captures the user's string from an `EditText`, dispatches it to the background thread executor to compute its embedding via `EmbeddingEngine`, and queries the database via `RagManager`.
- **Navigation**: Caches the returned chunks into an in-memory `List<String>` and manages index state (`currentChunkIndex`) to let users traverse back and forth through the retrieved context.

---

## 5. UI and Auxiliary Components

*(Note: These files handle layout boilerplate and UI state navigation. They do not contain backend RAG logic and can be skipped if you are purely focused on the ML/Database implementation.)*
- **`MainActivity.java`**: Orchestrates the core views. It queries a `SharedPreferences` flag (`is_rag_created`) to determine whether the database has content and conditionally locks/unlocks the "RAG Extraction" tab.
- **`TabsWrapper.java` & `TabConfig.java`**: Custom UI wrappers implementing standard Android `ViewPager2` and `TabLayoutMediator` behavior to construct the tab navigation.
- **`ModelDescriptor.java`**: A simple POJO containing metadata for downloadable embedding models (Model Name, ONNX Download URL, Vocab Download URL).