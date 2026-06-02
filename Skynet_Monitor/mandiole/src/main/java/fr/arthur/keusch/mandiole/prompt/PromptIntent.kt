package fr.arthur.keusch.mandiole.prompt

internal sealed class PromptIntent {
    data class QA(val systemPrompt: String? = null) : PromptIntent()
}
