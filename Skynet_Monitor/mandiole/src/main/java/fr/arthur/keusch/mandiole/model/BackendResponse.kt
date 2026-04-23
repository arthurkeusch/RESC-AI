package fr.arthur.keusch.mandiole.model

data class BackendResponse(
    val text: String,
    val thinkingText: String? = null,
    val tokenCount: Int? = null
)
