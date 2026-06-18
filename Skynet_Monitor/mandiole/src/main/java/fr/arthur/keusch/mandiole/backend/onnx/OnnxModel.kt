package fr.arthur.keusch.mandiole.backend.onnx

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxTensorLike
import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.util.Log
import fr.arthur.keusch.mandiole.util.createFloat16Tensor
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.nio.ShortBuffer

internal class OnnxModel(
    private val modelFile: File,
    private val config: ModelConfig
) {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession = initializeModel()
    private val sessionInputNames: List<String> = session.inputNames.toList()
    private val pastKeyValueSpecs: List<PastKeyValueSpec> = discoverPastKeyValueSpecs()
    private var unit: String = "CPU"
    private val isClosed = java.util.concurrent.atomic.AtomicBoolean(false)

    val executionUnit: String
        get() = unit

    companion object {
        const val MAX_TOKENS = 1024
        const val MAX_INPUT_TOKENS = 512
        const val TEMPERATURE = 0.8f
        const val REPETITION_PENALTY = 1.5f
        private const val TAG = "LLM"
    }

    private data class PastKeyValueSpec(
        val name: String,
        val shape: LongArray,
        val type: OnnxJavaType
    )

    fun close() {
        if (isClosed.compareAndSet(false, true)) {
            session.close()
        }
    }

    private fun initializeModel(): OrtSession {
        Log.d(TAG, "Loading model from: ${modelFile.absolutePath}")
        val opts = OrtSession.SessionOptions()
        try {
            runCatching {
                opts.addNnapi()
                Log.i(TAG, "NNAPI acceleration enabled")
                unit = "GPU (NNAPI)"
            }.onFailure {
                Log.w(TAG, "NNAPI not available, falling back to CPU: ${it.message}")
                unit = "CPU"
            }

            val session = env.createSession(modelFile.absolutePath, opts)
            Log.d(TAG, "Model loaded and session initialized")
            return session
        } finally {
            opts.close()
        }
    }

    private fun discoverPastKeyValueSpecs(): List<PastKeyValueSpec> {
        val inputInfo = session.getInputInfo()
        val specs = session.inputNames
            .filter { it.startsWith("past_key_values.") }
            .map { inputName ->
                val info = inputInfo[inputName]?.info as? TensorInfo
                    ?: throw IllegalStateException("Missing tensor info for input '$inputName'.")

                PastKeyValueSpec(
                    name = inputName,
                    shape = normalizePastKeyValueShape(info.shape),
                    type = info.type
                )
            }

        val expectedLayers = specs.size / 2
        if (expectedLayers != config.numLayers) {
            Log.w(
                TAG,
                "Model input schema exposes $expectedLayers KV-cache layers, " +
                        "but config.numLayers=${config.numLayers}. Using the model schema."
            )
        }

        return specs
    }

    private fun normalizePastKeyValueShape(rawShape: LongArray): LongArray {
        val normalized = rawShape.copyOf()
        if (normalized.isEmpty()) return normalized

        normalized[0] = normalized[0].takeIf { it > 0 } ?: config.batchSize.toLong()

        for (index in 1 until normalized.size) {
            if (normalized[index] > 0) continue

            normalized[index] = when {
                index == normalized.lastIndex -> config.headDim.toLong()
                index == 1 && normalized.size >= 4 -> config.numKvHeads.toLong()
                else -> 0L
            }
        }

        if (normalized.none { it == 0L } && normalized.size >= 3) {
            normalized[normalized.lastIndex - 1] = 0L
        }

        return normalized
    }

    private fun buildRunInputs(
        inputTensor: OnnxTensor,
        attentionTensor: OnnxTensor,
        posTensor: OnnxTensor,
        pastKeyValues: Map<String, OnnxTensor>
    ): LinkedHashMap<String, OnnxTensorLike> {
        val inputs = linkedMapOf<String, OnnxTensorLike>()

        sessionInputNames.forEach { inputName ->
            when (inputName) {
                "input_ids" -> inputs[inputName] = inputTensor
                "attention_mask" -> inputs[inputName] = attentionTensor
                "position_ids" -> inputs[inputName] = posTensor
                else -> {
                    val cacheTensor = pastKeyValues[inputName]
                    if (cacheTensor != null) {
                        inputs[inputName] = cacheTensor
                    }
                }
            }
        }

        return inputs
    }

    // Temperature scaling for logits
    private fun applyTemperature(logits: FloatArray, temperature: Float): FloatArray {
        if (temperature == 1.0f) return logits
        Log.d(TAG, "Applying temperature: $temperature")
        return FloatArray(logits.size) { i -> logits[i] / temperature }
    }

    private fun applyRepetitionPenalty(
        logits: FloatArray,
        generated: List<Int>,
        penalty: Float
    ): FloatArray {
        if (penalty == 1.0f) return logits
        Log.d(TAG, "Applying repetition penalty: $penalty")
        val adjusted = logits.copyOf()
        for (tokenId in generated) {
            if (tokenId in adjusted.indices) {
                if (adjusted[tokenId] < 0) {
                    adjusted[tokenId] *= penalty
                } else {
                    adjusted[tokenId] /= penalty
                }
            }
        }
        return adjusted
    }

    fun runInference(
        inputIds: IntArray,
        maxTokens: Int = MAX_TOKENS,
        endTokenId: Int = 151645
    ): IntArray {
        val generated = inputIds.toMutableList()

        for (i in 0 until maxTokens) {
            val seqLen = generated.size.toLong()
            Log.d(TAG, "Iteration $i | Sequence length: $seqLen")

            val inputTensor = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(generated.map { it.toLong() }.toLongArray()),
                longArrayOf(1, seqLen)
            )
            val attnTensor = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(LongArray(seqLen.toInt()) { 1L }),
                longArrayOf(1, seqLen)
            )
            val posTensor = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(LongArray(seqLen.toInt()) { it.toLong() }),
                longArrayOf(1, seqLen)
            )

            val results = session.run(
                mapOf(
                    "input_ids" to inputTensor,
                    "attention_mask" to attnTensor,
                    "position_ids" to posTensor
                )
            )

            val logits = (results[0].value as Array<Array<FloatArray>>)[0].last()
            val nextTokenId = logits.indices.maxByOrNull { logits[it] } ?: 0
            generated.add(nextTokenId)

            Log.d(TAG, "Generated token: $nextTokenId")

            inputTensor.close(); attnTensor.close(); posTensor.close(); results.close()
            if (nextTokenId == endTokenId) break
        }

        return generated.toIntArray()
    }

    fun runInferenceStreaming(
        inputIds: IntArray,
        maxTokens: Int = MAX_TOKENS,
        endTokenIds: Set<Int> = setOf(151645),
        shouldStop: () -> Boolean = { false },
        onTokenGenerated: (Int) -> Unit
    ) {
        val generated = inputIds.toMutableList()

        for (i in 0 until maxTokens) {
            if (shouldStop()) {
                Log.d(TAG, "Generation stopped early at token $i")
                break
            }

            val seqLen = generated.size.toLong()
            val inputIdsTensor = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(generated.map { it.toLong() }.toLongArray()),
                longArrayOf(1, seqLen)
            )
            val attnTensor = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(LongArray(seqLen.toInt()) { 1L }),
                longArrayOf(1, seqLen)
            )
            val posTensor = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(LongArray(seqLen.toInt()) { it.toLong() }),
                longArrayOf(1, seqLen)
            )

            val results = session.run(
                mapOf(
                    "input_ids" to inputIdsTensor,
                    "attention_mask" to attnTensor,
                    "position_ids" to posTensor
                )
            )

            val rawLogits = (results[0].value as Array<Array<FloatArray>>)[0].last()

            val nextTokenId = rawLogits.indices.maxByOrNull { rawLogits[it] } ?: 0
            generated.add(nextTokenId)

            Log.d(TAG, "Streaming token: $nextTokenId")
            inputIdsTensor.close(); attnTensor.close(); posTensor.close(); results.close()

            onTokenGenerated(nextTokenId)
            if (nextTokenId in endTokenIds) break
        }
    }

    fun runInferenceStreamingWithPastKV(
        inputIds: IntArray,
        maxTokens: Int = MAX_TOKENS,
        maxInputTokens: Int = MAX_INPUT_TOKENS,

        endTokenIds: Set<Int> = config.eosTokenIds,
        shouldStop: () -> Boolean = { false },
        onTokenGenerated: (Int) -> Unit
    ) {
        val promptTokens = if (inputIds.size > maxInputTokens) {
            Log.w(
                TAG,
                "Prompt had ${inputIds.size} tokens; truncated to last $maxInputTokens tokens."
            )
            inputIds.takeLast(maxInputTokens).toIntArray()
        } else {
            inputIds
        }
        val generated = promptTokens.toMutableList()

        val isQwen3 = config.modelName.contains("qwen3", ignoreCase = true)

        // Initialize empty past key/value cache for all layers
        val pastKeyValues = mutableMapOf<String, OnnxTensor>()
        pastKeyValueSpecs.forEach { spec ->
            val emptyKV = FloatArray(0)
            pastKeyValues[spec.name] = when (spec.type) {
                OnnxJavaType.FLOAT16 -> createFloat16Tensor(env, emptyKV, spec.shape)
                OnnxJavaType.FLOAT -> OnnxTensor.createTensor(
                    env,
                    FloatBuffer.wrap(emptyKV),
                    spec.shape
                )

                else -> throw IllegalArgumentException(
                    "Unsupported KV-cache tensor type for ${spec.name}: ${spec.type}"
                )
            }
        }

        var totalPosition = 0L

        val prefillChunkSize = 32
        var prefillCursor = 0

        while (prefillCursor < promptTokens.size) {
            val remaining = promptTokens.size - prefillCursor
            val currentChunkSize = if (remaining > prefillChunkSize) prefillChunkSize else remaining
            val currentChunk =
                promptTokens.sliceArray(prefillCursor until (prefillCursor + currentChunkSize))
            val isLastChunk = prefillCursor + currentChunkSize == promptTokens.size

            val seqLen = currentChunk.size.toLong()
            val inputTensor = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(currentChunk.map { it.toLong() }.toLongArray()),
                longArrayOf(1, seqLen)
            )

            val attentionTensor = run {
                val totalLen = (totalPosition + seqLen).toInt()
                val attn = LongArray(totalLen) { 1L }
                OnnxTensor.createTensor(
                    env,
                    LongBuffer.wrap(attn),
                    longArrayOf(1, totalLen.toLong())
                )
            }

            val posArray = LongArray(seqLen.toInt()) { j -> totalPosition + j }
            val posTensor =
                OnnxTensor.createTensor(env, LongBuffer.wrap(posArray), longArrayOf(1, seqLen))

            val inputs = buildRunInputs(inputTensor, attentionTensor, posTensor, pastKeyValues)
            val results = session.run(inputs)

            try {
                if (isLastChunk) {
                    val logitsTensor = results[0] as OnnxTensor
                    val shape = logitsTensor.info.shape
                    val vocabSize = shape[2].toInt()

                    val byteBuffer = logitsTensor.byteBuffer
                    byteBuffer.order(java.nio.ByteOrder.nativeOrder())
                    val floatBuffer = byteBuffer.asFloatBuffer()

                    val lastTokenLogits = FloatArray(vocabSize)
                    floatBuffer.position((shape[1].toInt() - 1) * vocabSize)
                    floatBuffer.get(lastTokenLogits)

                    val nextTokenId =
                        lastTokenLogits.indices.maxByOrNull { lastTokenLogits[it] } ?: 0
                    if (nextTokenId in endTokenIds) {
                        pastKeyValues.values.forEach { it.close() }
                        inputTensor.close(); attentionTensor.close(); posTensor.close(); results.close()
                        return
                    }

                    onTokenGenerated(nextTokenId)
                    generated.add(nextTokenId)
                }

                results.drop(1).forEachIndexed { index, result ->
                    val layer = index / 2
                    val kv = if (index % 2 == 0) "key" else "value"
                    val name = "past_key_values.$layer.$kv"
                    (result.value as? OnnxTensor)?.let {
                        val tensorCopy = cloneTensor(it)
                        pastKeyValues[name]?.close()
                        pastKeyValues[name] = tensorCopy
                    }
                }
            } finally {
                inputTensor.close(); attentionTensor.close(); posTensor.close(); results.close()
            }

            totalPosition += seqLen
            prefillCursor += currentChunkSize
        }

        for (i in 1 until maxTokens) {
            if (shouldStop()) break

            val currentInput = intArrayOf(generated.last())
            val seqLen = currentInput.size.toLong()

            val inputTensor = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(currentInput.map { it.toLong() }.toLongArray()),
                longArrayOf(1, seqLen)
            )

            val attentionTensor = run {
                val totalLen = (totalPosition + 1).toInt()
                val attn = LongArray(totalLen) { 1L }
                OnnxTensor.createTensor(
                    env,
                    LongBuffer.wrap(attn),
                    longArrayOf(1, totalLen.toLong())
                )
            }

            val posTensor = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(longArrayOf(totalPosition)),
                longArrayOf(1, 1)
            )

            val inputs = buildRunInputs(inputTensor, attentionTensor, posTensor, pastKeyValues)
            val results = session.run(inputs)

            try {
                val logitsTensor = results[0] as OnnxTensor
                val shape = logitsTensor.info.shape
                val vocabSize = shape[2].toInt()

                val byteBuffer = logitsTensor.byteBuffer
                byteBuffer.order(java.nio.ByteOrder.nativeOrder())
                val lastTokenLogits = FloatArray(vocabSize)
                byteBuffer.asFloatBuffer().get(lastTokenLogits)

                val nextTokenId =
                    lastTokenLogits.indices.maxByOrNull { lastTokenLogits[it] } ?: break
                if (nextTokenId in endTokenIds) break

                onTokenGenerated(nextTokenId)
                generated.add(nextTokenId)
                totalPosition += 1

                results.drop(1).forEachIndexed { index, result ->
                    val layer = index / 2
                    val kv = if (index % 2 == 0) "key" else "value"
                    val name = "past_key_values.$layer.$kv"
                    (result.value as? OnnxTensor)?.let {
                        val tensorCopy = cloneTensor(it)
                        pastKeyValues[name]?.close()
                        pastKeyValues[name] = tensorCopy
                    }
                }
            } finally {
                inputTensor.close(); attentionTensor.close(); posTensor.close(); results.close()
            }
        }

        pastKeyValues.values.forEach { it.close() }
    }

    private fun cloneTensor(source: OnnxTensor): OnnxTensor {
        val info = source.info as TensorInfo
        val shape = info.shape
        return when (info.type) {
            OnnxJavaType.FLOAT16 -> {
                val buffer = source.shortBuffer ?: throw IllegalStateException("Missing buffer")
                val copy = ShortArray(buffer.remaining())
                buffer.get(copy)
                OnnxTensor.createTensor(env, ShortBuffer.wrap(copy), shape, OnnxJavaType.FLOAT16)
            }

            OnnxJavaType.FLOAT -> {
                val buffer = source.floatBuffer ?: throw IllegalStateException("Missing buffer")
                val copy = FloatArray(buffer.remaining())
                buffer.get(copy)
                OnnxTensor.createTensor(env, FloatBuffer.wrap(copy), shape)
            }

            else -> throw IllegalArgumentException("Unsupported cached tensor type: ${info.type}")
        }
    }
}
