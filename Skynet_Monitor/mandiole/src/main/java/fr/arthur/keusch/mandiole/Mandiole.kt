package fr.arthur.keusch.mandiole

import android.content.Context
import fr.arthur.keusch.mandiole.backend.litert.GemmaLiteRtBackend
import fr.arthur.keusch.mandiole.backend.litert.QwenLiteRtBackend
import fr.arthur.keusch.mandiole.backend.onnx.OnnxChatBackend
import fr.arthur.keusch.mandiole.model.*
import fr.arthur.keusch.mandiole.util.ModelFileResolver
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * Main entry point for the Mandiole LLM library.
 */
class Mandiole(private val context: Context) : AutoCloseable {

    private val modelFileResolver = ModelFileResolver(context)
    private val backendLock = Mutex()
    private var activeBackend: fr.arthur.keusch.mandiole.backend.ChatBackend? = null

    /**
     * Represents a single turn in a chat conversation.
     */
    interface ChatTurn {
        val text: String
        val isUser: Boolean
        val thinkingText: String?
        val thinkingDurationMillis: Long?
    }

    /**
     * Represents the response from the LLM.
     */
    interface Response {
        val text: String
        val thinkingText: String?
        val tokenCount: Int?
    }

    /**
     * Represents a model's properties.
     */
    interface ModelDescriptor {
        val id: String
        val displayName: String
        val supportsThinking: Boolean
        val backendLabel: String
        val sizeLabel: String
        val deviceRecommendation: String
        val approxDownloadBytes: Long
    }

    /**
     * Information about a model download progress.
     */
    interface DownloadProgress {
        val fileName: String
        val bytesDownloaded: Long
        val totalBytes: Long?
    }

    /**
     * Name of the current hardware execution unit (e.g., "CPU", "GPU").
     */
    val executionUnit: String
        get() = activeBackend?.executionUnit ?: "None"

    /**
     * Checks if the model files are available locally.
     */
    fun isModelAvailable(descriptor: ModelDescriptor): Boolean {
        val internalDescriptor = ModelRegistry.findById(descriptor.id) ?: return false
        return modelFileResolver.isModelAvailable(internalDescriptor)
    }

    /**
     * Downloads the required files for the specified model.
     */
    suspend fun downloadModel(
        descriptor: ModelDescriptor,
        onProgress: (DownloadProgress) -> Unit
    ) {
        val internalDescriptor =
            ModelRegistry.findById(descriptor.id) ?: throw IllegalArgumentException("Unknown model")
        val downloader = fr.arthur.keusch.mandiole.download.ModelDownloader(modelFileResolver)
        downloader.downloadModel(internalDescriptor) { progress ->
            onProgress(object : DownloadProgress {
                override val fileName = progress.fileName
                override val bytesDownloaded = progress.bytesDownloaded
                override val totalBytes = progress.totalBytes
            })
        }
    }

    /**
     * Deletes the local files of the specified model.
     */
    fun deleteModel(descriptor: ModelDescriptor): Boolean {
        val internalDescriptor = ModelRegistry.findById(descriptor.id) ?: return true
        return modelFileResolver.deleteModelFiles(internalDescriptor)
    }

    /**
     * Loads the model into memory. Closes any previous model.
     */
    suspend fun loadModel(descriptor: ModelDescriptor) = backendLock.withLock {
        closeInternal()
        val internalDescriptor =
            ModelRegistry.findById(descriptor.id) ?: throw IllegalArgumentException("Unknown model")
        val backend = when (internalDescriptor) {
            is OnnxQwenSpec -> OnnxChatBackend(context, internalDescriptor, modelFileResolver)
            is QwenLiteRtSpec -> QwenLiteRtBackend(context, internalDescriptor, modelFileResolver)
            is GemmaLiteRtSpec -> GemmaLiteRtBackend(context, internalDescriptor, modelFileResolver)
        }
        backend.initialize()
        activeBackend = backend
    }

    /**
     * Generates a streaming reply based on the chat history.
     */
    suspend fun streamReply(
        history: List<ChatTurn>,
        thinkingEnabled: Boolean = true,
        onPartial: (Response) -> Unit
    ): Response = backendLock.withLock {
        val backend = activeBackend ?: throw IllegalStateException("No model loaded.")
        val internalHistory = history.map { turn ->
            ChatTurn(
                role = if (turn.isUser) ChatRole.USER else ChatRole.ASSISTANT,
                text = turn.text
            )
        }
        val result = backend.streamReply(internalHistory, thinkingEnabled) { resp ->
            onPartial(object : Response {
                override val text = resp.text
                override val thinkingText = resp.thinkingText
                override val tokenCount = resp.tokenCount
            })
        }
        return object : Response {
            override val text = result.text
            override val thinkingText = result.thinkingText
            override val tokenCount = result.tokenCount
        }
    }

    /**
     * Resets the conversation state with the given history.
     */
    suspend fun resetConversation(history: List<ChatTurn>, thinkingEnabled: Boolean) =
        backendLock.withLock {
            val internalHistory = history.map { turn ->
                ChatTurn(
                    role = if (turn.isUser) ChatRole.USER else ChatRole.ASSISTANT,
                    text = turn.text
                )
            }
            activeBackend?.resetConversation(internalHistory, thinkingEnabled)
        }

    /**
     * Interrupts the current generation process.
     */
    fun cancelGeneration() {
        activeBackend?.cancelGeneration()
    }

    /**
     * Releases all resources and closes the active model.
     */
    override fun close() {
        closeInternal()
    }

    private fun closeInternal() {
        synchronized(this) {
            activeBackend?.close()
            activeBackend = null
        }
    }

    /**
     * Creates a Turn object representing a user message.
     */
    fun userTurn(text: String): ChatTurn = createTurn(text, true)

    /**
     * Creates a Turn object representing an assistant message.
     */
    fun assistantTurn(text: String): ChatTurn = createTurn(text, false)

    private fun createTurn(text: String, isUser: Boolean) = object : ChatTurn {
        override val text = text
        override val isUser = isUser
        override val thinkingText = null
        override val thinkingDurationMillis = null
    }

    companion object {
        /**
         * Returns a list of all registered models.
         */
        fun getAllModels(): List<ModelDescriptor> = ModelRegistry.all.map { it as ModelDescriptor }
    }
}
