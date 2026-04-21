package fr.arthur.keusch.mandiole.model

data class RoleTokenIds(
    val systemStart: List<Int>,     // Tokens prepended before system prompt
    val userStart: List<Int>,       // Tokens prepended before user message
    val assistantStart: List<Int>,  // Tokens prepended before model/assistant response
    val endToken: Int               // Token appended at the end of each role block
)
