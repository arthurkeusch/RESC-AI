## X. The RAG (Retrieval-Augmented Generation) Module

As a preliminary note, it is important to clarify that this module does not implement a complete RAG system; it specifically handles the "Retrieval" component. The artificial intelligence responsible for the text "Generation" (the LLM) is not included in this particular module.

Originally, this module was conceptualized to assist firefighters during live interventions by instantly providing them (and/or an AI) with precise procedural excerpts required for specific emergency situations. However, this primary use case was ultimately abandoned due to field concerns. Firefighters expressed apprehension regarding the total response time (especially once the generation phase would be added) and feared potential AI hallucinations or errors during critical, life-threatening moments. Nevertheless, given the advanced state of development, the framework was finalized to serve alternative, highly valuable situations, such as pre-intervention planning and the analysis of training exercises.

### X.1. Integration, Development Environment, and Pipeline Overview

To embed this capability within the host application, the RAG framework is structured as an isolated local module (imported via `:rag` within the application's `build.gradle.kts` configuration).

#### Environment Prerequisites & Core Libraries
Resuming or modifying this native architecture requires strict environment configuration:

* **Application Permissions**: Dynamic downloading of the embedding models requires network access. The final host application's `AndroidManifest.xml` must declare the Internet permission.

```xml
<manifest>
    <uses-permission android:name="android.permission.INTERNET" />
</manifest>
```

* **Minimum SDK Target**: Set imperatively to `minSdk = 26` to comply with the underlying Hugging Face Deep Java Library (DJL) tokenization binaries.
* **Native C++ Compilation**: A native C++ structure must be declared via `CMakeLists.txt` using the compilation argument `-DANDROID_STL=c++_shared`. This ensures that the native shared libraries needed by the Hugging Face Tokenizer are safely linked, preventing runtime segmentation faults on the Android system.
* **Dependency Stack**: The module relies on *OkHttp* for remote asset downloading, *PDFBox-Android* for document parsing, *ONNX Runtime* for local model inference, and *ObjectBox* for specialized vector storage.

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

#### The Core RAG Pipeline
The data lifecycle within the module follows a rigid, sequential pipeline designed to be highly modular and easy to maintain:

```text
[ PDF Document Input ] ➔ [ PDFExtractor: Parsing & Cleaning ] ➔ [ Text Segmentation & Chunking ]
                                                                             │
[ Local Vector Store (ObjectBox) ] 🪚 [ Embedding Engine (ONNX) ] ➔ [ DJL Tokenizer Processing ]
              │
    [ Semantic Query K-NN Search ]
```

### X.2. Internal Architecture and Component Breakdown

The `RAG.java` class serves as the central orchestrator and the primary public gateway for external applications. While developers can technically interact with lower-level sub-components directly, `RAG.java` acts as a unified facade designed to offer a simple yet comprehensive API. It exposes key methods that control the entire life cycle:

* **`RAG.init(Context, RAGCallback, [String repoId])`**: Asynchronously boots the database, checks for local assets, and initializes the inference engine. The third parameter, `repoId`, is optional; it allows developers to define a custom Hugging Face model and forces the system to re-download the model if specified.
* **`RAG.inflateRAG(InputStream, RAGCallback)`**: Accepts a document file stream, passing it down the ingestion pipeline to parse, chunk, embed, and commit the data to the vector index.
* **`RAG.queryRAG(String)`**: Tokenizes a user's prompt, extracts its embedding vector, queries the database, and returns an array of matching text strings.

*Note: A `RAG.emptyRAG()` method is also available. It deletes all stored documents from the database, which is useful for fully resetting the vector index.*

#### Functional Class Segregation
* **Data Extraction (`PDFExtractor.java`)**: Leverages *PDFBox* to clean text files. It implements Java's native `BreakIterator` to dynamically discover sentence boundaries. It groups sentences into blocks based on a strict character limit, and maintain a structural context overlap between adjacent chunks to prevent the loss of semantic data.
* **Tokenization & Inference (`Tokenizer.java` & `EmbeddingEngine.java`)**: The `Tokenizer` instantiates the Hugging Face component via DJL. The `EmbeddingEngine` encapsulates the *ONNX Runtime* session, evaluating token tensors, applying *mean pooling*, and enforcing L2 normalization to output the finalized embedding vector.
* **Vector Database Persistence (`Item.java` & `ObjectBox.java`)**: `Item.java` defines the ObjectBox database schema. Crucially, the vector field is explicitly annotated with `@HnswIndex(type = VectorDistanceType.COSINE)`, configuring ObjectBox to maintain a graph index optimized for Cosine similarity K-Nearest Neighbor (K-NN) searches.

#### Implementation Cautions and Constraints
When modifying the module, developers must pay close attention to several structural constraints:
* **Fixed Embedding Dimensions**: The vector size is fixed by the currently selected model and defined in the configuration. If the embedding dimension is changed in `Config.java` to accommodate a new model, the application must be fully reinstalled (clearing app data), as the ObjectBox schema cannot dynamically resize existing vector columns.
* **Token Limits**: The maximum token count is intentionally kept relatively low to ensure compatibility with a wide variety of models.
* **Chunk Sizing Margins**: Because chunks are split based on a character count (not a token count) and strictly terminate at the end of a sentence, developers must leave a generous margin when defining the chunk size compared to the model's token limit to avoid truncation errors during inference.

### X.3. Evaluation and Performance (Benchmarking)

The benchmarking module is a decoupled diagnostic suite built to systematically assess the execution efficiency of various open-source embedding models across different contexts. This framework profiles execution times across the sequential steps of the RAG pipeline, and also returns the extracted chunks for each test-case.

#### Architecture & Test Structuring (`Benchmark.java`)
The evaluation framework is constructed around a nested data matrix designed to allow future developers to easily swap out or append new evaluation criteria. The test layout is managed via three core structural data classes:
* **`ModelBenchmark`**: Encapsulates the specific Hugging Face repository identifier being profiled, as well as the associated documents to test.
* **`DocumentBenchmark`**: Holds the descriptive name, the file source streams, and the corresponding array of test prompts.
* **`PromptBenchmark`**: Stores individual text queries designed to challenge the vector index.

*Optimization Note: If multiple models are testing the exact same documents, developers can define the `DocumentBenchmark[]` array once and reuse the exact same instance across different models. The same logic applies to reusing `PromptBenchmark[]` arrays across different documents.*

The benchmarking logic tracks state via `BenchmarkState` and executes recursively. When initiated, it loops through every model to initialize the target architecture, reads the input documents to build the vector index, and executes the semantic search for each distinct prompt. For every step, it precisely measures the elapsed time in milliseconds via system hardware clocks.

The output—including precise timing and the actual arrays of retrieved text chunks—is written directly into a JSON report. To assess the quality of the retrieved chunks (the actual relevance of the text output), the JSON must be analyzed post-execution. This analysis can be done manually by a human, processed by an external LLM to judge output acceptability, or evaluated using a larger, highly precise embedding model.

#### Execution Flow (`BenchmarkActivity.java`)
The benchmarking routine is built as a standalone Android application that fires automatically upon launch (`onCreate`).

```text
[ App Launch ] ➔ [ Asynchronous Document Download ] ➔ [ Define Model Matrix ] 
                                                                │
[ Export results.json ] 🏚 [ UI Real-Time Updates ] ➔ [ Execute Recursive Benchmarks ]
```

On launch, the activity boots a background thread that downloads the designated validation PDFs from remote web endpoints. Once the documents are buffered, it defines the test elements to evaluate, runs the recursive tests, and updates the progress in real-time on the UI. When execution terminates, it compiles a complete JSON result file and immediately provides it to the user via Android's `FileProvider`.

### X.4. User Interface (UI) Structure

The application's graphical architecture emphasizes a clean view separation, non-blocking background task scheduling, and real-time state reporting.

#### View Separation & App Lifecycle
* **`InitActivity.java`**: Acts as the startup controller. It handles the initial boot sequence by processing `RAG.init()` inside a background `ExecutorService`. Upon a successful setup, it securely forwards the user to the workspace; if an exception is caught, it shifts to an isolated `ErrorActivity.java` screen.
* **`RAGActivity.java`**: Represents the primary application workspace. It implements a decoupled tab view structure using an Android `ViewPager2` bound together with a `TabLayoutMediator`. On instantiation, it fires a lightweight, blank test query to evaluate index state. If a valid index is discovered, it automatically steers the user to the Query workspace; otherwise, it defaults focus onto the Build view.

#### Asynchronous Invocations and Real-Time Feedback
To maintain complete UI fluidness during processor-intensive tasks, both functional fragments strictly offload RAG interactions to dedicated background threads while communicating progress updates back to the main thread.

```text
[ BuildFragment UI ] ➔ Pick PDF ➔ [ Executor (Thread) ] ➔ RAG.inflateRAG() ➔ [ Callback Updates UI ]
[ QueryFragment UI ] ➔ Input    ➔ [ Executor (Thread) ] ➔ RAG.queryRAG()   ➔ [ Posts Pagination View ]
```

* **The Build Interface (`BuildFragment.java`)**: Provides an entry screen containing a file selection trigger. When a local PDF is chosen, the file URI stream is handed off to an executor that calls `RAG.inflateRAG()`. As the orchestrator processes the document, step notifications are pushed through UI thread updates.
* **The Query Interface (`QueryFragment.java`)**: Contains a text entry component and an execution action trigger. When a search is requested, the input string is packaged and sent to evaluate `RAG.queryRAG()`. To prevent text layout overflow on mobile screens, the fragment manages data through structured navigation buttons ("Previous" and "Next"), allowing users to page cleanly through the individual matching text chunks.