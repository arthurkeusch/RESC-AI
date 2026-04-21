package resc.ai.skynetmonitor.viewmodel

import android.annotation.SuppressLint
import android.app.ActivityManager
import android.app.Application
import android.content.Context
import android.util.Log
import androidx.compose.runtime.mutableStateOf
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import fr.arthur.keusch.mandiole.Mandiole
import fr.arthur.keusch.mandiole.backend.ChatBackend
import fr.arthur.keusch.mandiole.model.ChatRole
import fr.arthur.keusch.mandiole.model.ChatTurn
import fr.arthur.keusch.mandiole.model.ModelDescriptor
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import resc.ai.skynetmonitor.service.DeviceInfoService
import resc.ai.skynetmonitor.service.DownloadState

data class ChatSessionState(
    val isRunning: Boolean = false,
    val modelName: String = "",
    val output: List<String> = emptyList()
)

class DeviceInfoViewModel(application: Application) : AndroidViewModel(application) {

    val ctx: Context get() = getApplication<Application>().applicationContext
    private val mandiole = Mandiole(ctx)

    private val _remoteModels = MutableStateFlow<List<ModelDescriptor>>(emptyList())
    val remoteModels: StateFlow<List<ModelDescriptor>> = _remoteModels.asStateFlow()

    private val _downloadState = MutableStateFlow<DownloadState?>(null)
    val downloadState: StateFlow<DownloadState?> = _downloadState.asStateFlow()

    private val _isDeleting = MutableStateFlow(false)
    val isDeleting: StateFlow<Boolean> = _isDeleting.asStateFlow()

    private val _lastDeleteCompleted = MutableStateFlow<String?>(null)
    val lastDeleteCompleted: StateFlow<String?> = _lastDeleteCompleted.asStateFlow()

    var hardwareInfo = mutableStateOf<Map<String, String>>(emptyMap())
        private set
    var systemState = mutableStateOf<Map<String, String>>(emptyMap())
        private set
    var historyData = mutableStateOf<Map<String, List<Float>>>(emptyMap())
        private set

    private val _chat = MutableStateFlow(ChatSessionState())
    val benchmarkState: StateFlow<ChatSessionState> = _chat.asStateFlow()

    private var downloadJob: Job? = null
    private var currentBackend: ChatBackend? = null
    private val chatHistory = mutableListOf<ChatTurn>()

    init {
        hardwareInfo.value = DeviceInfoService.getStaticHardwareInfo(ctx)
        viewModelScope.launch {
            while (true) {
                val newState = DeviceInfoService.getDynamicSystemState(ctx)
                systemState.value = newState
                val updatedHistory = historyData.value.toMutableMap()
                newState.forEach { (key, rawValue) ->
                    val numeric = rawValue
                        .replace(",", ".")
                        .replace(Regex("[^0-9.]"), "")
                        .toFloatOrNull()
                    if (numeric != null) {
                        val list = updatedHistory[key]?.toMutableList() ?: mutableListOf()
                        list.add(numeric)
                        if (list.size > 60) list.removeAt(0)
                        updatedHistory[key] = list
                    }
                }
                historyData.value = updatedHistory
                delay(1000L)
            }
        }
    }

    fun isModelLocal(model: ModelDescriptor): Boolean {
        return mandiole.isModelAvailable(model)
    }

    fun loadModelsRemote() {
        _remoteModels.value = mandiole.getAvailableModels()
    }

    fun downloadModel(model: ModelDescriptor) {
        downloadJob?.cancel()
        downloadJob = viewModelScope.launch {
            _downloadState.value = DownloadState(
                name = model.displayName,
                bytesReceived = 0L,
                totalBytes = model.approxDownloadBytes,
                speedBytesPerSec = 0L,
                etaSeconds = -1L,
                progress = 0
            )
            try {
                mandiole.downloadModel(model) { progress ->
                    val received = progress.bytesDownloaded
                    val total = progress.totalBytes ?: model.approxDownloadBytes
                    val p = if (total > 0) ((received * 100) / total).toInt() else 0
                    
                    _downloadState.value = _downloadState.value?.copy(
                        bytesReceived = received,
                        totalBytes = total,
                        progress = p.coerceIn(0, 100)
                    )
                }
                _downloadState.value = _downloadState.value?.copy(
                    progress = 100,
                    etaSeconds = 0,
                    speedBytesPerSec = 0
                )
            } catch (ce: CancellationException) {
                _downloadState.value = null
                throw ce
            } catch (e: Exception) {
                Log.e("ViewModel", "Download failed", e)
                _downloadState.value = _downloadState.value?.copy(etaSeconds = -1)
            } finally {
                downloadJob = null
            }
        }
    }

    fun cancelDownload() {
        downloadJob?.cancel()
        _downloadState.value = null
        downloadJob = null
    }

    fun clearDownloadState() {
        _downloadState.value = null
    }

    fun deleteLocalModel(model: ModelDescriptor) {
        viewModelScope.launch {
            try {
                _isDeleting.value = true
                mandiole.deleteModel(model)
                _lastDeleteCompleted.value = model.id
                loadModelsRemote()
            } finally {
                _isDeleting.value = false
            }
        }
    }

    fun consumeDeleteEvent() {
        _lastDeleteCompleted.value = null
    }

    fun startBenchmark(model: ModelDescriptor) {
        viewModelScope.launch {
            try {
                _chat.value = ChatSessionState(
                    isRunning = true,
                    modelName = model.displayName,
                    output = listOf("Loading model ${model.displayName}...")
                )
                
                if (!mandiole.isModelAvailable(model)) {
                    _chat.value = _chat.value.copy(output = listOf("Downloading model..."))
                    downloadModel(model)
                    return@launch
                }

                currentBackend?.close()
                val backend = mandiole.loadModel(model)
                currentBackend = backend
                chatHistory.clear()
                
                _chat.value = _chat.value.copy(
                    output = listOf("✅ Model loaded: ${model.displayName}")
                )
            } catch (e: Exception) {
                Log.e("Benchmark failed", "Error", e)
                _chat.value = _chat.value.copy(
                    isRunning = false,
                    output = _chat.value.output + "Error: ${e.message}"
                )
            }
        }
    }

    fun sendPrompt(prompt: String) {
        val userTurn = ChatTurn(role = ChatRole.USER, text = prompt)
        chatHistory.add(userTurn)
        
        val list = _chat.value.output.toMutableList()
        list.add("> $prompt")
        _chat.value = _chat.value.copy(output = list)
        
        viewModelScope.launch {
            try {
                val backend = currentBackend ?: return@launch
                backend.streamReply(chatHistory, thinkingEnabled = false) { response ->
                    // Stream update logic could go here
                }.also { finalResponse ->
                    chatHistory.add(ChatTurn(role = ChatRole.ASSISTANT, text = finalResponse.text))
                    val out = _chat.value.output.toMutableList()
                    out.add(finalResponse.text)
                    _chat.value = _chat.value.copy(output = out)
                }
            } catch (e: Exception) {
                val out = _chat.value.output.toMutableList()
                out.add("Error: ${e.message}")
                _chat.value = _chat.value.copy(output = out)
            }
        }
    }

    fun stopBenchmark() {
        currentBackend?.close()
        currentBackend = null
        _chat.value = _chat.value.copy(isRunning = false)
    }

    @SuppressLint("DefaultLocale")
    fun formatSize(bytes: Long): String {
        if (bytes <= 0) return "—"
        val kb = bytes / 1024.0
        val mb = kb / 1024.0
        val gb = mb / 1024.0
        return when {
            gb >= 1 -> String.format("%.2f GB", gb)
            mb >= 1 -> String.format("%.2f MB", mb)
            else -> String.format("%.2f KB", kb)
        }
    }

    fun getBoundsFor(key: String): Pair<Float, Float> {
        return when {
            key.contains("Battery Temperature", ignoreCase = true) -> 0f to 50f
            key.contains("Battery Level", ignoreCase = true) -> 0f to 100f
            key.contains("Temp", ignoreCase = true) -> 0f to 100f
            key.contains("RAM", ignoreCase = true) -> {
                val am = ctx.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
                val memInfo = ActivityManager.MemoryInfo()
                am.getMemoryInfo(memInfo)
                val total = (memInfo.totalMem / (1024 * 1024 * 1024)).toFloat()
                0f to total
            }

            else -> 0f to 100f
        }
    }
}
