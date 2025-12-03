package resc.ai.skynetmonitor.service

data class RemoteModel(
    val id: Long,
    val name: String,
    val filename: String,
    val sizeBytes: Long,
    val params: String = "",
    val isLocal: Boolean = false
)
