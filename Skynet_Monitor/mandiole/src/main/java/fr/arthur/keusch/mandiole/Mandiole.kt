package fr.arthur.keusch.mandiole

import android.content.Context
import fr.arthur.keusch.mandiole.backend.ChatBackend
import fr.arthur.keusch.mandiole.backend.litert.GemmaLiteRtBackend
import fr.arthur.keusch.mandiole.backend.litert.QwenLiteRtBackend
import fr.arthur.keusch.mandiole.backend.onnx.OnnxChatBackend
import fr.arthur.keusch.mandiole.download.ModelDownloader
import fr.arthur.keusch.mandiole.download.ModelDownloadProgress
import fr.arthur.keusch.mandiole.model.BackendResponse
import fr.arthur.keusch.mandiole.model.ChatRole
import fr.arthur.keusch.mandiole.model.ChatTurn
import fr.arthur.keusch.mandiole.model.GemmaLiteRtSpec
import fr.arthur.keusch.mandiole.model.ModelDescriptor
import fr.arthur.keusch.mandiole.model.ModelRegistry
import fr.arthur.keusch.mandiole.model.OnnxQwenSpec
import fr.arthur.keusch.mandiole.model.QwenLiteRtSpec
import fr.arthur.keusch.mandiole.model.asModelMemoryTurns
import fr.arthur.keusch.mandiole.util.ModelFileResolver

/**
 * Main entry point for the Mandiole LLM library.
 * This class acts as a facade to handle model discovery, downloading, and loading.
 */
class Mandiole(private val context: Context) {

    private val modelFileResolver = ModelFileResolver(context)
    private val modelDownloader = ModelDownloader(modelFileResolver)

    // --- Model Discovery & Management ---

    /**
     * Returns all registered model descriptors.
     */
    fun getAvailableModels(): List<ModelDescriptor> {
        return ModelRegistry.all
    }

    /**
     * Finds a model descriptor by its ID.
     */
    fun getModel(id: String): ModelDescriptor? {
        return ModelRegistry.findById(id)
    }

    /**
     * Checks if a model's files are available (either downloaded or in assets).
     */
    fun isModelAvailable(modelId: String): Boolean {
        val descriptor = getModel(modelId) ?: return false
        return modelFileResolver.isModelAvailable(descriptor)
    }

    /**
     * Checks if a model's files are available (either downloaded or in assets).
     */
    fun isModelAvailable(descriptor: ModelDescriptor): Boolean {
        return modelFileResolver.isModelAvailable(descriptor)
    }

    // --- Downloading ---

    /**
     * Downloads the required files for a model.
     */
    suspend fun downloadModel(
        modelId: String,
        onProgress: (ModelDownloadProgress) -> Unit
    ) {
        val descriptor = getModel(modelId) ?: throw IllegalArgumentException("Model not found: $modelId")
        downloadModel(descriptor, onProgress)
    }

    /**
     * Downloads the required files for a model.
     */
    suspend fun downloadModel(
        descriptor: ModelDescriptor,
        onProgress: (ModelDownloadProgress) -> Unit
    ) {
        modelDownloader.downloadModel(descriptor, onProgress)
    }

    // --- Deletion ---

    /**
     * Deletes the downloaded files for a model.
     */
    fun deleteModel(modelId: String): Boolean {
        val descriptor = getModel(modelId) ?: return true
        return deleteModel(descriptor)
    }

    /**
     * Deletes the downloaded files for a model.
     */
    fun deleteModel(descriptor: ModelDescriptor): Boolean {
        return modelFileResolver.deleteModelFiles(descriptor)
    }

    // --- Inference & Backend ---

    /**
     * Loads and initializes a model backend.
     */
    suspend fun loadModel(modelId: String): ChatBackend {
        val descriptor = getModel(modelId) ?: throw IllegalArgumentException("Model not found: $modelId")
        return loadModel(descriptor)
    }

    /**
     * Loads and initializes a model backend.
     */
    suspend fun loadModel(descriptor: ModelDescriptor): ChatBackend {
        val backend = when (descriptor) {
            is OnnxQwenSpec -> OnnxChatBackend(context, descriptor, modelFileResolver)
            is QwenLiteRtSpec -> QwenLiteRtBackend(context, descriptor, modelFileResolver)
            is GemmaLiteRtSpec -> GemmaLiteRtBackend(context, descriptor, modelFileResolver)
        }
        backend.initialize()
        return backend
    }

    /**
     * Utility to prepare chat history for inference by stripping UI-specific metadata.
     */
    fun prepareHistoryForInference(history: List<ChatTurn>): List<ChatTurn> {
        return history.asModelMemoryTurns()
    }

    /**
     * Creates a new chat turn.
     */
    fun createChatTurn(role: ChatRole, text: String): ChatTurn {
        return ChatTurn(role = role, text = text)
    }
}

// Re-export important types for easier access if they are in the same package or widely used
typealias MandioleChatBackend = ChatBackend
typealias MandioleDownloadProgress = ModelDownloadProgress
typealias MandioleResponse = BackendResponse
