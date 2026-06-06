The `build.gradle.kts` file must contain the following:

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

If it's used as a module, the `build.gradle.kts` file (from the app this time) must contain the following:

```kts
dependencies {
    implementation(project(":rag"))
}
```

The `AndroidManifest.xml` file must contain the following:

```xml
<manifest>
    <uses-permission android:name="android.permission.INTERNET" />
</manifest>
```

Notes:
- The minimum SDK version is set to 26, as it's the required minimum for "*Hugging Face (DJL)*".
- The `INTERNET` permission is necessary for downloading the model from Hugging Face.
- The used libraries are:
    - `OkHttp` for handling HTTP requests to download the model from Hugging Face.
    - `Hugging Face (DJL)` for tokenization.
    - `ObjectBox` for database management.
    - `ONNXRuntime (Android)` for running the embedding model.
    - `PDFBox (Android)` for extracting text from PDF files.
- A dummy C++ project must be created in `src/main/cpp` to include libraries required by the Tokenizer. There is alternative solutions, but this one is the simplest.