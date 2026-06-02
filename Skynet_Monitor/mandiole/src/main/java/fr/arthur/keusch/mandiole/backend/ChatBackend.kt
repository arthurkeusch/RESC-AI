package fr.arthur.keusch.mandiole.backend

import fr.arthur.keusch.mandiole.model.ChatTurn
import fr.arthur.keusch.mandiole.model.BackendResponse

internal interface ChatBackend : AutoCloseable {
    val executionUnit: String
    suspend fun initialize()
    suspend fun resetConversation(history: List<ChatTurn>, thinkingEnabled: Boolean)
    suspend fun streamReply(
        history: List<ChatTurn>,
        thinkingEnabled: Boolean,
        onPartial: (BackendResponse) -> Unit
    ): BackendResponse

    fun cancelGeneration()
}
