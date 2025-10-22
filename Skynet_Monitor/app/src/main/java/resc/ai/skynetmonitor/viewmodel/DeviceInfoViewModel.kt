package resc.ai.skynetmonitor.viewmodel

import android.util.Log
import android.annotation.SuppressLint
import android.app.ActivityManager
import android.app.Application
import android.content.Context
import androidx.compose.runtime.mutableStateOf
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import resc.ai.skynetmonitor.service.*
import java.io.File

data class ChatSessionState(
    val isRunning: Boolean = false,
    val modelName: String = "",
    val output: List<String> = emptyList()
)

class DeviceInfoViewModel(application: Application) : AndroidViewModel(application) {

    val ctx: Context get() = getApplication<Application>().applicationContext

    private val _remoteModels = MutableStateFlow<List<RemoteModel>>(emptyList())
    val remoteModels: StateFlow<List<RemoteModel>> = _remoteModels.asStateFlow()

    private val _downloadState = MutableStateFlow<DownloadState?>(null)
    val downloadState: StateFlow<DownloadState?> = _downloadState.asStateFlow()

    private val _isDeleting = MutableStateFlow(false)
    val isDeleting: StateFlow<Boolean> = _isDeleting.asStateFlow()

    private val _lastDeleteCompleted = MutableStateFlow<String?>(null)
    val lastDeleteCompleted: StateFlow<String?> = _lastDeleteCompleted.asStateFlow()

    private val metaByName = mutableMapOf<String, RemoteModel>()

    var hardwareInfo = mutableStateOf<Map<String, String>>(emptyMap())
        private set
    var systemState = mutableStateOf<Map<String, String>>(emptyMap())
        private set
    var historyData = mutableStateOf<Map<String, List<Float>>>(emptyMap())
        private set

    private val _chat = MutableStateFlow(ChatSessionState())
    val benchmarkState: StateFlow<ChatSessionState> = _chat.asStateFlow()

    private var downloadJob: Job? = null
    private var currentDownload: RemoteModel? = null

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
        loadModelsRemote()
    }

    fun loadModelsRemote() {
        viewModelScope.launch {
            try {
                val remotes = ModelService.fetchRemoteModels().map { remote ->
                    remote.copy(isLocal = ModelService.isModelDownloaded(ctx, remote.filename))
                }
                metaByName.clear()
                remotes.forEach { metaByName[it.name] = it }
                _remoteModels.value = remotes
            } catch (_: Exception) {
            }
        }
    }

    fun onUseRemote(remote: RemoteModel) {
        val path = ModelService.getLocalModelPath(ctx, remote.filename)
        startBenchmark(path)
    }

    fun downloadModel(remote: RemoteModel) {
        downloadJob?.cancel()
        currentDownload = remote
        downloadJob = viewModelScope.launch {
            _downloadState.value = DownloadState(
                name = remote.name,
                bytesReceived = 0L,
                totalBytes = remote.sizeBytes,
                speedBytesPerSec = 0L,
                etaSeconds = -1L,
                progress = 0
            )
            try {
                ModelService.downloadModel(ctx, remote) { st ->
                    _downloadState.value = st
                }
                _downloadState.value = _downloadState.value?.copy(
                    progress = 100,
                    etaSeconds = 0,
                    speedBytesPerSec = 0
                )
            } catch (ce: CancellationException) {
                currentDownload?.let {
                    val target = ModelService.resolveLocalFile(ctx, it.filename)
                    val tmp = File(target.parentFile, "${target.name}.part")
                    if (tmp.exists()) tmp.delete()
                }
                _downloadState.value = null
                throw ce
            } catch (_: Exception) {
                _downloadState.value = _downloadState.value?.copy(etaSeconds = -1)
            } finally {
                currentDownload = null
                downloadJob = null
            }
        }
    }

    fun cancelDownload() {
        downloadJob?.cancel()
        currentDownload?.let {
            val target = ModelService.resolveLocalFile(ctx, it.filename)
            val tmp = File(target.parentFile, "${target.name}.part")
            if (tmp.exists()) tmp.delete()
        }
        _downloadState.value = null
        currentDownload = null
        downloadJob = null
    }

    fun clearDownloadState() {
        _downloadState.value = null
    }

    fun deleteLocalModel(remote: RemoteModel) {
        viewModelScope.launch {
            try {
                _isDeleting.value = true
                val f = ModelService.resolveLocalFile(ctx, remote.filename)
                if (f.exists()) f.delete()
                _lastDeleteCompleted.value = remote.filename
            } finally {
                _isDeleting.value = false
            }
        }
    }

    fun consumeDeleteEvent() {
        _lastDeleteCompleted.value = null
    }

    fun startBenchmark(modelPath: String) {
        viewModelScope.launch {
            try {
                _chat.value = ChatSessionState(
                    isRunning = true,
                    modelName = modelPath.substringAfterLast("/"),
                    output = emptyList()
                )
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
                        _chat.value = ChatSessionState(
                            isRunning = false,
                            modelName = _chat.value.modelName,
                            output = _chat.value.output + "Error: $it"
                        )
                    }
                )
            } catch (e: Error) {
                e.stackTrace.forEach { stackTraceElement ->
                    Log.e("Benchmark failed", "    $stackTraceElement")
                }
                throw e
            }
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
