package fr.arthur.keusch.mandiole.backend.onnx

import android.content.Context
import fr.arthur.keusch.mandiole.backend.ChatBackend
import fr.arthur.keusch.mandiole.model.BackendResponse
import fr.arthur.keusch.mandiole.model.ChatTurn
import fr.arthur.keusch.mandiole.model.OnnxQwenSpec
import fr.arthur.keusch.mandiole.model.RoleTokenIds
import fr.arthur.keusch.mandiole.parser.QwenResponseParser
import fr.arthur.keusch.mandiole.prompt.PromptBuilder
import fr.arthur.keusch.mandiole.prompt.PromptIntent
import fr.arthur.keusch.mandiole.tokenizer.BpeTokenizer
import fr.arthur.keusch.mandiole.util.ModelFileResolver
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicBoolean

class OnnxChatBackend(
    private val context: Context,
    private val spec: OnnxQwenSpec,
    private val modelFileResolver: ModelFileResolver
) : ChatBackend {

    private lateinit var tokenizer: BpeTokenizer
    private lateinit var config: ModelConfig
    private lateinit var promptBuilder: PromptBuilder
    private lateinit var onnxModel: OnnxModel
    private val cancelRequested = AtomicBoolean(false)

    override val executionUnit: String
        get() = if (::onnxModel.isInitialized) onnxModel.executionUnit else "CPU"

    override suspend fun initialize() = withContext(Dispatchers.IO) {
        tokenizer = BpeTokenizer(context, spec, modelFileResolver)
        config = spec.toModelConfig(tokenizer)
        promptBuilder = PromptBuilder(tokenizer, config)

        val modelFile = modelFileResolver.resolveModelFile(spec)
        onnxModel = OnnxModel(modelFile, config)
    }

    override suspend fun resetConversation(history: List<ChatTurn>, thinkingEnabled: Boolean) {
        cancelRequested.set(false)
    }

    override suspend fun streamReply(
        history: List<ChatTurn>,
        thinkingEnabled: Boolean,
        onPartial: (BackendResponse) -> Unit
    ): BackendResponse = withContext(Dispatchers.IO) {
        cancelRequested.set(false)
        val coroutineIsActive = { isActive }
        val isQwen3 = spec.modelName.equals("qwen3", ignoreCase = true)

        val systemPrompt = buildSystemPrompt(thinkingEnabled)
        val promptTokens = promptBuilder.buildPromptTokens(history, PromptIntent.QA(systemPrompt))
        val responseBuilder = StringBuilder()
        val streamDecoder = tokenizer.createStreamDecoder()
        var tokenCount = 0

        onnxModel.runInferenceStreamingWithPastKV(
            inputIds = promptTokens,
            endTokenIds = config.eosTokenIds,
            shouldStop = { cancelRequested.get() || !coroutineIsActive() },
            onTokenGenerated = { tokenId ->
                val tokenText = streamDecoder.append(tokenId)
                tokenCount++

                responseBuilder.append(tokenText)
                onPartial(parseBackendResponse(responseBuilder.toString(), isQwen3, tokenCount))
            }
        )

        val trailingText = streamDecoder.flush()
        if (trailingText.isNotEmpty()) {
            responseBuilder.append(trailingText)
            onPartial(parseBackendResponse(responseBuilder.toString(), isQwen3, tokenCount))
        }

        parseBackendResponse(responseBuilder.toString(), isQwen3, tokenCount)
    }

    override fun cancelGeneration() {
        cancelRequested.set(true)
    }

    override fun close() {
        if (::onnxModel.isInitialized) {
            onnxModel.close()
        }
    }

    private fun buildSystemPrompt(thinkingEnabled: Boolean): String {
        val isQwen3 = spec.modelName.contains("qwen3", ignoreCase = true)
        if (!isQwen3 || !spec.thinkingModeAvailable) {
            return spec.defaultSystemPrompt
        }

        return when {
            // Mode manuel : On donne une indication claire mais on laisse le LLM maitre
            !thinkingEnabled -> "${spec.defaultSystemPrompt} /no_think"
            else -> "${spec.defaultSystemPrompt} /think"
        }
    }

    private fun parseBackendResponse(rawOutput: String, isQwen3: Boolean, tokenCount: Int): BackendResponse {
        return if (isQwen3) {
            val base = QwenResponseParser.parseVisibleResponse(rawOutput)
            base.copy(tokenCount = tokenCount)
        } else {
            BackendResponse(text = rawOutput, tokenCount = tokenCount)
        }
    }

    private fun OnnxQwenSpec.toModelConfig(tokenizer: BpeTokenizer): ModelConfig {
        val roleTokens = RoleTokenIds(
            systemStart = listOf(
                tokenizer.getTokenId("<|im_start|>"),
                tokenizer.getTokenId("system"),
                tokenizer.getTokenId("Ċ")
            ),
            userStart = listOf(
                tokenizer.getTokenId("<|im_start|>"),
                tokenizer.getTokenId("user"),
                tokenizer.getTokenId("Ċ")
            ),
            assistantStart = listOf(
                tokenizer.getTokenId("<|im_start|>"),
                tokenizer.getTokenId("assistant"),
                tokenizer.getTokenId("Ċ")
            ),
            endToken = tokenizer.getTokenId("<|im_end|>")
        )

        return ModelConfig(
            modelName = modelName,
            modelPath = modelAssetName,
            promptStyle = promptStyle,
            eosTokenIds = eosTokenIds,
            numLayers = numLayers,
            numKvHeads = numKvHeads,
            headDim = headDim,
            batchSize = batchSize,
            defaultSystemPrompt = defaultSystemPrompt,
            roleTokenIds = roleTokens,
            scalarPosId = scalarPosId,
            dtype = dtype,
            IsThinkingModeAvailable = thinkingModeAvailable
        )
    }
}
