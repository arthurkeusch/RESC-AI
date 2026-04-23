package fr.arthur.keusch.mandiole.backend.litert

import android.content.Context
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import fr.arthur.keusch.mandiole.backend.ChatBackend
import fr.arthur.keusch.mandiole.model.BackendResponse
import fr.arthur.keusch.mandiole.model.ChatRole
import fr.arthur.keusch.mandiole.model.ChatTurn
import fr.arthur.keusch.mandiole.model.GemmaLiteRtSpec
import fr.arthur.keusch.mandiole.util.ModelFileResolver
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.withContext
import java.io.File

class GemmaLiteRtBackend(
    private val context: Context,
    private val spec: GemmaLiteRtSpec,
    private val modelFileResolver: ModelFileResolver
) : ChatBackend {

    companion object {
        private const val THOUGHT_CHANNEL_NAME = "thought"
    }

    private lateinit var engine: Engine
    private var conversation: Conversation? = null
    private var unit: String = "CPU"

    override val executionUnit: String
        get() = unit

    override suspend fun initialize() = withContext(Dispatchers.IO) {
        val modelFile = modelFileResolver.resolveModelFile(spec)

        // Détection de la présence d'OpenCL sur l'appareil.
        val isOpenClAvailable = listOf(
            "/system/lib64/libOpenCL.so",
            "/system/vendor/lib64/libOpenCL.so",
            "/vendor/lib64/libOpenCL.so",
            "/vendor/lib64/egl/libGLES_mali.so", // Souvent là sur Samsung
            "/system/vendor/lib/libOpenCL.so",
            "/system/lib/libOpenCL.so"
        ).any { File(it).exists() }

        // Détection plus large pour inclure Samsung, Mali, Exynos et Mediatek (mt)
        val isSamsung = android.os.Build.MANUFACTURER.contains("samsung", ignoreCase = true)
        val isMaliOrExynos = android.os.Build.HARDWARE.contains("mali", ignoreCase = true) || 
                             android.os.Build.BOARD.contains("exynos", ignoreCase = true) ||
                             android.os.Build.HARDWARE.contains("mt", ignoreCase = true) // Mediatek

        // On évite le GPU sur Samsung pour le moment car LiteRT-LM y a des problèmes de chargement OpenCL
        val useGpu = isOpenClAvailable && !isMaliOrExynos && !isSamsung

        Log.d("LLM", "Init: isOpenCl=$isOpenClAvailable, isMali=$isMaliOrExynos, isSamsung=$isSamsung, useGpu=$useGpu")

        val gpuResult = if (useGpu) {
            Log.i("LLM", "Attempting GPU initialization...")
            runCatching {
                Engine(
                    EngineConfig(
                        modelPath = modelFile.absolutePath,
                        backend = Backend.GPU(),
                        cacheDir = context.cacheDir.absolutePath
                    )
                ).apply { initialize() }
            }.onSuccess { unit = "GPU" }
        } else {
            Result.failure(Exception(if (!isOpenClAvailable) "OpenCL unavailable" else "Forced CPU for stability on Mali/Exynos"))
        }

        engine = gpuResult.getOrElse { gpuError ->
            Log.w("LLM", "Fallback to CPU mode: ${gpuError.message}")
            unit = "CPU"
            runCatching {
                Engine(
                    EngineConfig(
                        modelPath = modelFile.absolutePath,
                        backend = Backend.CPU(),
                        cacheDir = context.cacheDir.absolutePath
                    )
                ).apply { initialize() }
            }.getOrElse { cpuError ->
                throw IllegalStateException(
                    "Failed to initialize LiteRT-LM GPU (${gpuError.message}) and CPU (${cpuError.message}).",
                    cpuError
                )
            }
        }
    }

    override suspend fun resetConversation(history: List<ChatTurn>, thinkingEnabled: Boolean) {
        recreateConversation(history, thinkingEnabled)
    }

    override suspend fun streamReply(
        history: List<ChatTurn>,
        thinkingEnabled: Boolean,
        onPartial: (BackendResponse) -> Unit
    ): BackendResponse = withContext(Dispatchers.IO) {
        require(history.isNotEmpty() && history.last().role == ChatRole.USER) {
            "Gemma backend expects the final history turn to be the user's prompt."
        }

        val initialHistory = history.dropLast(1)
        val userTurn = history.last()
        recreateConversation(initialHistory, thinkingEnabled)

        val activeConversation = conversation
            ?: throw IllegalStateException("Conversation was not created.")

        val textBuilder = StringBuilder()
        val thinkingBuilder = StringBuilder()
        var chunkCount = 0

        activeConversation.sendMessageAsync(userTurn.text).collect { message ->
            val chunkText = message.contents.toString()
            if (chunkText.isNotEmpty()) {
                textBuilder.append(chunkText)
                chunkCount++
            }

            val thoughtChunk = message.channels[THOUGHT_CHANNEL_NAME].orEmpty()
            if (thoughtChunk.isNotEmpty()) {
                thinkingBuilder.append(thoughtChunk)
                // thinking chunks are separate from content usually
            }

            onPartial(
                BackendResponse(
                    text = textBuilder.toString(),
                    thinkingText = thinkingBuilder.toString().takeIf { it.isNotBlank() },
                    tokenCount = chunkCount
                )
            )
        }

        BackendResponse(
            text = textBuilder.toString(),
            thinkingText = thinkingBuilder.toString().takeIf { it.isNotBlank() },
            tokenCount = chunkCount
        )
    }

    override fun cancelGeneration() {
        conversation?.cancelProcess()
    }

    override fun close() {
        closeConversation()
        runCatching {
            if (::engine.isInitialized) {
                engine.close()
            }
        }
    }

    private fun recreateConversation(history: List<ChatTurn>, thinkingEnabled: Boolean) {
        closeConversation()
        conversation = engine.createConversation(
            ConversationConfig(
                systemInstruction = Contents.of(buildSystemInstruction(thinkingEnabled)),
                initialMessages = history.map { turn ->
                    when (turn.role) {
                        ChatRole.USER -> Message.user(turn.text)
                        ChatRole.ASSISTANT -> Message.model(turn.text)
                    }
                },
                channels = if (thinkingEnabled) null else emptyList()
            )
        )
    }

    private fun buildSystemInstruction(thinkingEnabled: Boolean): String {
        return if (thinkingEnabled) {
            "<|think|>\n${spec.defaultSystemInstruction}"
        } else {
            spec.defaultSystemInstruction
        }
    }

    private fun closeConversation() {
        val currentConversation = conversation ?: return
        runCatching { currentConversation.close() }
        conversation = null
    }
}
