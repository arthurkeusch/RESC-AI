# Technical Documentation: RAG (Retrieval-Augmented Generation) on Android

This document is the technical `README.md` of the Android RAG module. It explains how to integrate this module into an application, how to configure a development environment to modify it, and details the internal architecture of the various Java scripts.

---

## 1. Implementation in a third-party application (Usage)

To use this RAG module in your Android application, you must import it as a module (for example, named `:rag`). Then, you just need to declare the dependency in the `build.gradle.kts` file **of your host application**:

```kts
dependencies {
    implementation(project(":rag"))
}
```

You can then use the main public class `RAG.java` to initialize the system (`RAG.init()`), create the index from a PDF file, or execute your semantic searches via `RAG.queryRAG()`.

---

## 2. Preparing the environment for modification

If you need to resume, modify, or recompile this source module, the environment requires strict configuration, mainly related to the C++ dependencies (used by the artificial intelligence for tokenization and inference).

### Application permissions
Dynamic downloading of the embedding model from Hugging Face requires Internet access. Ensure that the module's `AndroidManifest.xml` file contains the following permission:

```xml
<manifest>
    <uses-permission android:name="android.permission.INTERNET" />
</manifest>
```

### Module Gradle configuration (`build.gradle.kts`)
Here is the complete configuration required for the module to compile correctly. 

**Crucial points:**
* **Minimum SDK**: The `minSdk` version must imperatively be set to `26`, as it is a prerequisite for the Hugging Face (DJL) library.
* **C++ Compatibility**: A "dummy" C++ project must be declared via a `CMakeLists.txt` file located in `src/main/cpp`. The `-DANDROID_STL=c++_shared` argument allows including the C++ shared libraries required by the native Tokenizer so that it does not crash at runtime.

```kts
plugins {
    id("io.objectbox:5.4.2")
}

android {
    defaultConfig {
        minSdk = 26

        externalNativeBuild {
            cmake {
                arguments("-DANDROID_STL=c++_shared")
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
        }
    }
}

dependencies {
    implementation("com.squareup.okhttp3:okhttp:5.3.2")
    annotationProcessor("io.objectbox:objectbox-processor:5.4.2")
    implementation("ai.djl.huggingface:tokenizers:0.36.0")
    runtimeOnly("ai.djl.android:tokenizer-native:0.36.0")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.26.0")
    implementation("com.tom-roush:pdfbox-android:2.0.27.0")
    
    implementation("androidx.annotation:annotation-jvm:1.10.0")
}
```

### Libraries used
* **OkHttp**: Allows managing HTTP requests for downloading the model (and the tokenizer) from the Hugging Face servers.
* **Hugging Face (DJL)**: Used for text tokenization (splitting into IDs compatible with the AI model).
* **ObjectBox**: An extremely fast local database, used here as a Vector Database (Vector DB) to store embeddings and perform semantic similarity searches.
* **ONNXRuntime (Android)**: The inference engine allowing the local execution of the ONNX model in order to generate the embedding vectors.
* **PDFBox (Android)**: Used to read, analyze, and extract all the text from the provided PDF files.

---

## 3. Architecture and internal operation of the code

The RAG module code is organized modularly. Here is the logical explanation of each file's role:

### Configuration
* **`Config.java`**: Centralizes all the constant variables of the application. Here we can find, for example, the Hugging Face repository ID to use by default (`REPO_ID`), the parameters to use for the model (`EMBEDDING_DIM`, `N_TOKENS`), and the output parameters (`CHUNK_SIZE`, `N_RESULTS`).

### Main Orchestration
* **`RAG.java`**: It is the orchestrator of the system. It provides an interface with key methods:
    * **`init()`**: Starts ObjectBox, checks the existence of local models, and triggers the download (via `HFDownloader`) if necessary before initializing the AI engines.
    * **`createRAG()`**: Coordinates reading a PDF, extracting its chunks, converting each chunk into an embedding, and storing it in the database.
    * **`queryRAG()`**: Tokenizes the user's question, requests the corresponding vector (embedding), and queries ObjectBox to retrieve the most relevant excerpts. A `RAGCallback` is used to handle progress and errors.

### Downloading the Models
* **`Downloader.java`**: A generic utility class based on *OkHttp*. It handles downloading a file from the internet to the device's local storage while notifying progress via callbacks (`onProgress`, `onSuccess`, `onError`).
* **`HFDownloader.java`**: Built on top of the `Downloader`. It specifically knows the structure of Hugging Face URLs and guarantees that the model (`model.onnx`) and its configuration file (`tokenizer.json`) are fully downloaded before giving the green light.

### Data Extraction and Preparation
* **`PDFExtractor.java`**: Leverages *PDFBox*. It cleans the raw text (handles line breaks and typographic hyphenation). It then uses Java's native `BreakIterator` to cleanly split the text by **sentences** and builds "chunks" (blocks of parameterized size). The algorithm deliberately includes an overlap between each chunk to preserve semantic context during the search.

### Embedding
* **`Tokenizer.java`**: Loads the native Hugging Face component via *DJL*. This class converts complex strings (text or questions) into raw numerical lists (`Tokens`) of a strict size dictated by `Config.N_TOKENS`. It returns the `inputIds`, `attentionMask`, and `tokenTypeIds`.
* **`EmbeddingEngine.java`**: Manages the *ONNXRuntime* session. This class takes the identifiers generated by the Tokenizer and submits them to the `model.onnx` model. It then applies *mean pooling* (a global average over the relevant vectors identified by the attention mask) and finalizes with L2 normalization. This makes it possible to obtain the "mathematical meaning" of the text (the embedding).

### Vector Database (Vector DB)
* **`Item.java`**: The entity (database table) of *ObjectBox*. It has a unique `ID`, a text (`chunk`), and a floating vector (`embedding`). Crucially, the embedding field uses the `@HnswIndex` annotation which tells ObjectBox to create a geometric similarity search index based on Cosine distance (`VectorDistanceType.COSINE`).
* **`ObjectBox.java`**: The database manager. Contains the `putItem()` method to insert the processed data and `getItems()` to launch the mathematical query that finds the `k` nearest semantic neighbors (K-NN) during a search.