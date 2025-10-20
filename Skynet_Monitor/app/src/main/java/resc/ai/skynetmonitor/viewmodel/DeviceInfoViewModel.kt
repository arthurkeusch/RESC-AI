package resc.ai.skynetmonitor.viewmodel

import android.annotation.SuppressLint
import android.app.ActivityManager
import android.app.Application
import android.content.Context
import android.util.Log
import androidx.compose.runtime.mutableStateOf
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import resc.ai.skynetmonitor.service.DeviceInfoService
import resc.ai.skynetmonitor.service.DownloadState
import resc.ai.skynetmonitor.service.ModelLauncherService
import resc.ai.skynetmonitor.service.ModelService
import resc.ai.skynetmonitor.service.RemoteModel
import resc.ai.skynetmonitor.service.ModelService.isModelDownloaded
import resc.ai.skynetmonitor.ui.components.ModelInfo

data class ChatSessionState(
    val isRunning: Boolean = false,
    val modelName: String = "",
    val output: List<String> = emptyList()
)

class DeviceInfoViewModel(application: Application) : AndroidViewModel(application) {
    val ctx: Context get() = getApplication<Application>().applicationContext

    private val _models = MutableStateFlow<List<ModelInfo>>(emptyList())
    val models: StateFlow<List<ModelInfo>> = _models

    private val _downloadState = MutableStateFlow<DownloadState?>(null)
    val downloadState: StateFlow<DownloadState?> = _downloadState

    private val metaByName = mutableMapOf<String, RemoteModel>()

    var hardwareInfo = mutableStateOf<Map<String, String>>(emptyMap())
        private set
    var systemState = mutableStateOf<Map<String, String>>(emptyMap())
        private set
    var historyData = mutableStateOf<Map<String, List<Float>>>(emptyMap())
        private set

    private val _chat = MutableStateFlow(ChatSessionState())
    val benchmarkState: StateFlow<ChatSessionState> = _chat.asStateFlow()

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

    fun setApiUrl(url: String) {
        ModelService.setApiUrl(url)
        loadModels()
    }

    fun loadModels() {
        viewModelScope.launch {
            _models.value = emptyList()
            try {
                var remotes = ModelService.fetchRemoteModels()
                metaByName.clear()
                remotes.forEach { metaByName[it.name] = it }
                _models.value = remotes.map {
                    ModelInfo(
                        name = it.name,
                        size = formatFileSize(it.sizeBytes),
                        parameters = it.params
                    )
                }
            } catch (_: Exception) {}
        }
    }

    fun selectModel(model: ModelInfo, onSelected: (ModelInfo) -> Unit) {
        val meta = metaByName[model.name] ?: return
        if (isModelDownloaded(ctx, meta.filename)) {
            onSelected(model)
            return
        }
        viewModelScope.launch {
            _downloadState.value = DownloadState(
                name = meta.name,
                bytesReceived = 0L,
                totalBytes = meta.sizeBytes,
                speedBytesPerSec = 0L,
                etaSeconds = -1L,
                progress = 0
            )
            try {
                ModelService.downloadModel(ctx, meta) { st ->
                    _downloadState.value = st
                }
                _downloadState.value = null
                onSelected(model)
            } catch (_: Exception) {
                _downloadState.value = null
            }
        }
    }

    fun startBenchmarkFor(model: ModelInfo) {
        val meta = metaByName[model.name] ?: return
        val path = ModelService.getLocalModelPath(ctx, meta.filename)
        startBenchmark(path)
    }

    fun startBenchmark(modelPath: String) {
        viewModelScope.launch {
            _chat.value = ChatSessionState(isRunning = true, modelName = modelPath.substringAfterLast("/"), output = emptyList())
            ModelLauncherService.startModel(
                context = ctx,
                modelPath = modelPath,
                onOutput = { line ->
                    val list = _chat.value.output.toMutableList()
                    list.add(line)
                    _chat.value = _chat.value.copy(output = list)
                },
                onReady = {},
                onError = {
                    _chat.value = ChatSessionState(isRunning = false, modelName = _chat.value.modelName, output = _chat.value.output + "Error: $it")
                }
            )
        }
    }

    fun sendPrompt(prompt: String) {
        val list = _chat.value.output.toMutableList()
        list.add("> $prompt")
        _chat.value = _chat.value.copy(output = list)
        ModelLauncherService.sendPrompt(prompt) { response ->
            val out = _chat.value.output.toMutableList()
            out.add(response)
            _chat.value = _chat.value.copy(output = out)
        }
    }

    fun stopBenchmark() {
        _chat.value = _chat.value.copy(isRunning = false)
    }

    @SuppressLint("DefaultLocale")
    private fun formatFileSize(bytes: Long): String {
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
                val total = (memInfo.totalMem.toDouble() / (1024 * 1024 * 1024)).toFloat()
                0f to total
            }
            else -> 0f to 100f
        }
    }
}
