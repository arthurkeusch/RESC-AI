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
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import resc.ai.skynetmonitor.service.DeviceInfoService
import resc.ai.skynetmonitor.service.DownloadState
import resc.ai.skynetmonitor.service.DatasetItem
import resc.ai.skynetmonitor.service.PromptService
import resc.ai.skynetmonitor.service.ModelService
import resc.ai.skynetmonitor.service.PerformanceSample

data class ChatMessage(
    val isUser: Boolean,
    val text: String,
    val thinkingText: String? = null,
    val thinkingDurationSeconds: Int? = null
)

data class ChatSessionState(
    val isRunning: Boolean = false,
    val isModelLoaded: Boolean = false,
    val isGenerating: Boolean = false,
    val isBenchmarking: Boolean = false,
    val thinkingEnabled: Boolean = false,
    val canThink: Boolean = false,
    val modelName: String = "",
    val executionUnit: String? = null,
    val messages: List<ChatMessage> = emptyList(),
    val currentStep: BenchmarkStep = BenchmarkStep.MODEL_SELECTION,
    val datasets: List<DatasetItem> = emptyList(),
    val selectedDatasetIds: Set<Int> = emptySet(),
    val currentDatasetIndex: Int = 0,
    val currentPromptIndex: Int = 0,
    val totalPromptsInSelectedDatasets: Int = 0,
    val showStatsPanel: Boolean = false
)

enum class BenchmarkStep {
    MODEL_SELECTION,
    DATASET_SELECTION,
    EXECUTING
}

class DeviceInfoViewModel(application: Application) : AndroidViewModel(application) {

    val ctx: Context get() = getApplication<Application>().applicationContext
    private val mandiole = Mandiole(ctx)

    private val _remoteModels = MutableStateFlow<List<Mandiole.ModelDescriptor>>(emptyList())
    val remoteModels: StateFlow<List<Mandiole.ModelDescriptor>> = _remoteModels.asStateFlow()

    private val _localModels = MutableStateFlow<List<Mandiole.ModelDescriptor>>(emptyList())

    val localModels: StateFlow<List<Mandiole.ModelDescriptor>> = _localModels.asStateFlow()

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

    private var deviceId: Int? = null
    private var downloadJob: Job? = null
    private var inferenceJob: Job? = null
    private var currentModel: Mandiole.ModelDescriptor? = null
    private val chatHistory = mutableListOf<Mandiole.ChatTurn>()

    init {
        hardwareInfo.value = DeviceInfoService.getStaticHardwareInfo(ctx)

        viewModelScope.launch {
            deviceId = PromptService.registerOrGetDevice(ctx, hardwareInfo.value)
        }

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

    fun isModelLocal(model: Mandiole.ModelDescriptor): Boolean {
        return mandiole.isModelAvailable(model)
    }

    fun loadModelsRemote() {
        val remoteList = Mandiole.getAllModels()
        _remoteModels.value = remoteList
        val localList = remoteList.filter { isModelLocal(it) }
        _localModels.value = localList
    }

    fun downloadModel(model: Mandiole.ModelDescriptor) {
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

                // If a model was being selected for chat/benchmark, load it now
                if (_chat.value.isRunning && !_chat.value.isModelLoaded && currentModel == model) {
                    if (_chat.value.isBenchmarking && _chat.value.currentStep == BenchmarkStep.EXECUTING) {
                        runBenchmark()
                    } else if (!_chat.value.isBenchmarking) {
                        startChat(model)
                    }
                }
            } catch (ce: CancellationException) {
                _downloadState.value = null
                throw ce
            } catch (e: Exception) {
                Log.e("LLM", "Download failed", e)
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

    fun deleteLocalModel(model: Mandiole.ModelDescriptor) {
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

    fun startChat(model: Mandiole.ModelDescriptor) {
        viewModelScope.launch {
            try {
                _chat.value = ChatSessionState(
                    isRunning = true,
                    isModelLoaded = false,
                    isBenchmarking = false,
                    thinkingEnabled = model.supportsThinking,
                    canThink = model.supportsThinking,
                    modelName = model.displayName,
                    currentStep = BenchmarkStep.EXECUTING,
                    messages = emptyList()
                )

                if (!mandiole.isModelAvailable(model)) {
                    downloadModel(model)
                    return@launch
                }

                currentModel = model
                mandiole.loadModel(model)

                if (!_chat.value.isRunning) {
                    mandiole.close()
                    return@launch
                }

                chatHistory.clear()

                _chat.value = _chat.value.copy(
                    isModelLoaded = true,
                    executionUnit = mandiole.executionUnit
                )
            } catch (e: Exception) {
                Log.e("LLM", "Error starting chat", e)
                _chat.value = _chat.value.copy(
                    isRunning = false,
                    messages = _chat.value.messages + ChatMessage(
                        isUser = false,
                        "Error: ${e.message}"
                    )
                )
            }
        }
    }

    fun startSimpleChatFlow() {
        _chat.value = ChatSessionState(
            isRunning = true,
            isBenchmarking = false,
            currentStep = BenchmarkStep.MODEL_SELECTION
        )
        loadModelsRemote()
    }

    fun startBenchmarkFlow() {
        _chat.value = ChatSessionState(
            isRunning = true,
            isBenchmarking = true,
            currentStep = BenchmarkStep.MODEL_SELECTION
        )
        loadModelsRemote()
    }

    fun selectModelForBenchmark(model: Mandiole.ModelDescriptor) {
        viewModelScope.launch {
            if (!_chat.value.isBenchmarking) {
                startChat(model)
                return@launch
            }

            val datasets = PromptService.fetchDatasets(ctx) ?: emptyList()
            _chat.update {
                it.copy(
                    modelName = model.displayName,
                    currentStep = BenchmarkStep.DATASET_SELECTION,
                    datasets = datasets,
                    canThink = model.supportsThinking,
                    thinkingEnabled = model.supportsThinking
                )
            }
            currentModel = model
        }
    }

    fun toggleDatasetSelection(datasetId: Int) {
        _chat.update { state ->
            val newSelected = if (state.selectedDatasetIds.contains(datasetId)) {
                state.selectedDatasetIds - datasetId
            } else {
                state.selectedDatasetIds + datasetId
            }
            state.copy(selectedDatasetIds = newSelected)
        }
    }

    fun toggleStatsPanel() {
        _chat.update { it.copy(showStatsPanel = !it.showStatsPanel) }
    }

    fun runBenchmark() {
        val model = currentModel ?: return
        val selectedIds = _chat.value.selectedDatasetIds
        val selectedDatasets = _chat.value.datasets.filter { selectedIds.contains(it.id) }

        if (selectedDatasets.isEmpty()) return

        val totalPrompts = selectedDatasets.sumOf { it.prompts.size }
        _chat.update {
            it.copy(
                currentStep = BenchmarkStep.EXECUTING,
                messages = emptyList(),
                totalPromptsInSelectedDatasets = totalPrompts,
                currentDatasetIndex = 0,
                currentPromptIndex = 0
            )
        }

        inferenceJob?.cancel()
        inferenceJob = viewModelScope.launch {
            try {
                if (!mandiole.isModelAvailable(model)) {
                    downloadModel(model)
                    return@launch
                }

                mandiole.loadModel(model)

                _chat.update {
                    it.copy(
                        isModelLoaded = true,
                        executionUnit = mandiole.executionUnit
                    )
                }

                val dbModelId = ModelService.registerOrGetModel(ctx, model.displayName) ?: 1L
                Log.d("LLM", "Benchmark model registered with ID: $dbModelId")

                var overallPromptCounter = 0
                for ((dIdx, dataset) in selectedDatasets.withIndex()) {
                    _chat.update { it.copy(currentDatasetIndex = dIdx) }
                    chatHistory.clear()

                    for ((pIdx, promptItem) in dataset.prompts.withIndex()) {
                        _chat.update { it.copy(currentPromptIndex = overallPromptCounter) }
                        if (!dataset.isConversational) {
                            chatHistory.clear()
                        }

                        val promptText = promptItem.prompt
                        chatHistory.add(mandiole.userTurn(promptText))

                        _chat.update { state ->
                            state.copy(
                                isGenerating = true,
                                messages = state.messages + ChatMessage(
                                    isUser = true,
                                    promptText
                                ) + ChatMessage(isUser = false, "")
                            )
                        }

                        val startTime = System.currentTimeMillis()
                        val isThinkingModeActive = _chat.value.thinkingEnabled
                        val performanceSamples = mutableListOf<PerformanceSample>()

                        val samplingJob = viewModelScope.launch {
                            while (coroutineContext.isActive) {
                                performanceSamples.add(
                                    DeviceInfoService.getCurrentPerformanceSample(
                                        ctx,
                                        startTime
                                    )
                                )
                                delay(1000L)
                            }
                        }

                        val response = mandiole.streamReply(
                            chatHistory,
                            thinkingEnabled = isThinkingModeActive
                        ) { partial ->
                            val currentTime = System.currentTimeMillis()
                            val durationSec = if (partial.thinkingText != null) {
                                ((currentTime - startTime) / 1000).toInt()
                            } else null

                            _chat.update { state ->
                                val newMessages = state.messages.toMutableList()
                                if (newMessages.isNotEmpty()) {
                                    newMessages[newMessages.size - 1] = ChatMessage(
                                        isUser = false,
                                        text = partial.text,
                                        thinkingText = partial.thinkingText,
                                        thinkingDurationSeconds = durationSec
                                    )
                                }
                                state.copy(messages = newMessages)
                            }
                        }

                        samplingJob.cancel()
                        val endTime = System.currentTimeMillis()
                        val durationMs = endTime - startTime
                        val tokenCount =
                            response.tokenCount ?: response.text.split(Regex("\\s+")).size
                        val tokensPerS =
                            if (durationMs > 0) (tokenCount.toFloat() / (durationMs / 1000.0f)) else 0f

                        val totalDurationSec = if (response.thinkingText != null) {
                            (durationMs / 1000).toInt()
                        } else null

                        chatHistory.add(mandiole.assistantTurn(response.text))

                        _chat.update { state ->
                            val newMessages = state.messages.toMutableList()
                            if (newMessages.isNotEmpty()) {
                                newMessages[newMessages.size - 1] = ChatMessage(
                                    isUser = false,
                                    text = response.text,
                                    thinkingText = response.thinkingText,
                                    thinkingDurationSeconds = totalDurationSec
                                )
                            }
                            val limitedMessages =
                                if (newMessages.size > 50) newMessages.takeLast(50) else newMessages
                            state.copy(messages = limitedMessages, isGenerating = false)
                        }

                        try {
                            PromptService.submitBenchmarkResult(
                                context = ctx,
                                response = response.text,
                                idPrompt = promptItem.id,
                                idModel = dbModelId,
                                idDevices = deviceId ?: 1,
                                isThink = response.thinkingText != null,
                                responseTimeMs = durationMs,
                                responseTokenCount = tokenCount,
                                responseTokensPerS = tokensPerS,
                                performanceSamples = performanceSamples
                            )
                        } catch (e: Exception) {
                            Log.e("LLM", "Failed to submit result", e)
                        }
                        overallPromptCounter++
                    }
                }
                _chat.update { it.copy(currentPromptIndex = overallPromptCounter) }
            } catch (e: Exception) {
                Log.e("LLM", "Benchmark execution failed", e)
                _chat.update { it.copy(isGenerating = false) }
            }
        }
    }

    fun setThinkingEnabled(enabled: Boolean) {
        _chat.update { it.copy(thinkingEnabled = enabled) }
    }

    fun sendPrompt(prompt: String) {
        val userTurn = mandiole.userTurn(prompt)
        chatHistory.add(userTurn)

        _chat.update { state ->
            state.copy(
                isGenerating = true,
                messages = state.messages + ChatMessage(
                    isUser = true,
                    prompt
                ) + ChatMessage(isUser = false, "")
            )
        }

        inferenceJob?.cancel()
        inferenceJob = viewModelScope.launch {
            try {
                val isThinkingModeActive = _chat.value.thinkingEnabled
                val startTime = System.currentTimeMillis()

                mandiole.streamReply(
                    chatHistory,
                    thinkingEnabled = isThinkingModeActive
                ) { response ->
                    val currentTime = System.currentTimeMillis()
                    val durationSec = if (response.thinkingText != null) {
                        ((currentTime - startTime) / 1000).toInt()
                    } else null

                    _chat.update { state ->
                        val newMessages = state.messages.toMutableList()
                        if (newMessages.isNotEmpty()) {
                            newMessages[newMessages.size - 1] = ChatMessage(
                                isUser = false,
                                text = response.text,
                                thinkingText = response.thinkingText,
                                thinkingDurationSeconds = durationSec
                            )
                        }
                        state.copy(messages = newMessages)
                    }
                }.also { finalResponse ->
                    val totalDurationSec = if (finalResponse.thinkingText != null) {
                        ((System.currentTimeMillis() - startTime) / 1000).toInt()
                    } else null

                    chatHistory.add(mandiole.assistantTurn(finalResponse.text))
                    _chat.update { state ->
                        val newMessages = state.messages.toMutableList()
                        if (newMessages.isNotEmpty()) {
                            newMessages[newMessages.size - 1] = ChatMessage(
                                isUser = false,
                                text = finalResponse.text,
                                thinkingText = finalResponse.thinkingText,
                                thinkingDurationSeconds = totalDurationSec
                            )
                        }
                        state.copy(messages = newMessages, isGenerating = false)
                    }
                }
            } catch (e: Exception) {
                _chat.update { state ->
                    state.copy(
                        isGenerating = false,
                        messages = state.messages + ChatMessage(
                            isUser = false,
                            "Error: ${e.message}"
                        )
                    )
                }
            }
        }
    }

    fun cancelGeneration() {
        mandiole.cancelGeneration()
        inferenceJob?.cancel()
        _chat.update { it.copy(isGenerating = false) }
    }

    fun stopBenchmark() {
        _chat.update { it.copy(isRunning = false, isGenerating = false) }
        mandiole.cancelGeneration()
        inferenceJob?.cancel()

        viewModelScope.launch(Dispatchers.IO) {
            try {
                inferenceJob?.join()
                mandiole.close()
            } catch (e: Exception) {
                Log.e("LLM", "Error closing backend", e)
            }
        }
    }

    @SuppressLint("DefaultLocale")
    fun formatSize(bytes: Long): String {
        if (bytes <= 0) return "—"
        val kb = bytes / 1024.0
        val mb = kb / 1024.0
        val gb = mb / 1024.0
        return when {
            gb >= 1 -> String.format("%.2f GB", gb)
            mb >= 1 -> String.format("%.1f MB", mb)
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

    override fun onCleared() {
        super.onCleared()
        mandiole.close()
    }
}
