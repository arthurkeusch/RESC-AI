package resc.ai.skynetmonitor.service

import android.content.Context
import java.io.File

object ModelLauncherService {

    fun startModel(
        context: Context,
        modelPath: String,
        onOutput: (String) -> Unit,
        onReady: () -> Unit,
        onError: (String) -> Unit
    ) {
        val file = File(modelPath)
        if (!file.exists()) {
            onError("Model file not found")
            return
        }
        when (file.extension.lowercase()) {
            "gguf" -> launchGgufModel(file, onOutput, onReady, onError)
            else -> onError("Unsupported model format: ${file.extension}")
        }
    }

    private fun launchGgufModel(
        file: File,
        onOutput: (String) -> Unit,
        onReady: () -> Unit,
        onError: (String) -> Unit
    ) {
        onReady()
        onOutput("Model loaded: ${file.name}")
        onOutput("You can start chatting.")
    }

    fun sendPrompt(prompt: String, onResponse: (String) -> Unit) {
        onResponse("Model response to: \"$prompt\"")
    }
}
