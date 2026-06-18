package fr.arthur.keusch.mandiole.model

internal data class RoleTokenIds(
    val systemStart: List<Int>,
    val userStart: List<Int>,
    val assistantStart: List<Int>,
    val endToken: Int
)
