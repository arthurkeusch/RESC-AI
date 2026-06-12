plugins {
    id("com.android.library")
    id("io.objectbox") version "5.4.2"
}

android {
    namespace = "com.example.anhilyx.rescai.rag"
    compileSdk {
        version = release(36) {
            minorApiLevel = 1
        }
    }

    defaultConfig {
        minSdk = 26

        externalNativeBuild {
            cmake {
                arguments("-DANDROID_STL=c++_shared")
            }
        }

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
        }
    }
}

//noinspection UseTomlInstead
dependencies {
    implementation("com.squareup.okhttp3:okhttp:5.3.2")
    annotationProcessor("io.objectbox:objectbox-processor:5.4.2")
    implementation("ai.djl.huggingface:tokenizers:0.33.0")
    runtimeOnly("ai.djl.android:tokenizer-native:0.33.0")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.26.0")
    implementation("com.tom-roush:pdfbox-android:2.0.27.0")

    implementation("androidx.annotation:annotation-jvm:1.10.0")

    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
}