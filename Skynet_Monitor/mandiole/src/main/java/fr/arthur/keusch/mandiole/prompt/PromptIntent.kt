package fr.arthur.keusch.mandiole.prompt

sealed class PromptIntent {
    data class QA(val systemPrompt: String? = null) : PromptIntent()
}
