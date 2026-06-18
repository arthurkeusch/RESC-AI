package fr.arthur.keusch.mandiole.backend.onnx

import fr.arthur.keusch.mandiole.model.RoleTokenIds

internal enum class PromptStyle {
    QWEN2_5,
    QWEN3
}

internal data class ModelConfig(
    val modelName: String,
    val modelPath: String = "model.onnx",
    val promptStyle: PromptStyle,
    val eosTokenIds: Set<Int>,
    val numLayers: Int,
    val numKvHeads: Int,
    val headDim: Int,
    val batchSize: Int,
    val defaultSystemPrompt: String,
    val roleTokenIds: RoleTokenIds,
    val scalarPosId: Boolean = false,
    val dtype: String = "float32",
    val IsThinkingModeAvailable: Boolean = false
)
