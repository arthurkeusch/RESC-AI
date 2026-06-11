package resc.ai.skynetmonitor.viewmodel

import android.annotation.SuppressLint
import android.app.ActivityManager
import android.app.Application
import android.content.Context
import android.util.Log
import android.widget.Toast
import androidx.compose.runtime.mutableStateOf
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.anhilyx.rescai.rag.RAG
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
    val thinkingDurationSeconds: Int? = null,
    val ragQuery: String? = null,
    val ragResults: List<String>? = null,
    val ragReasoning: String? = null,
    val ragStatus: RagStatus = RagStatus.IDLE
)

enum class RagStatus {
    IDLE,
    ANALYZING,
    SEARCHING,
    SYNTHESIZING,
    SUCCESS,
    NOT_NEEDED
}

data class ChatSessionState(
    val isRunning: Boolean = false,
    val isModelLoaded: Boolean = false,
    val isGenerating: Boolean = false,
    val isBenchmarking: Boolean = false,
    val thinkingEnabled: Boolean = false,
    val ragEnabled: Boolean = false,
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
    val totalInputTokens: Int = 0,
    val processedInputTokens: Int = 0,
    val totalOutputTokens: Int = 0,
    val showStatsPanel: Boolean = false,
    val benchmarkElapsedSeconds: Long = 0L,
    val benchmarkRemainingSeconds: Long? = null
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

    private val _tpsHistory = MutableStateFlow<List<Float>>(emptyList())
    val tpsHistory: StateFlow<List<Float>> = _tpsHistory.asStateFlow()

    private val _contextUsageHistory = MutableStateFlow<List<Float>>(emptyList())
    val contextUsageHistory: StateFlow<List<Float>> = _contextUsageHistory.asStateFlow()

    private val _maxObservedTps = MutableStateFlow(1f)
    val maxObservedTps: StateFlow<Float> = _maxObservedTps.asStateFlow()

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
        _tpsHistory.value = emptyList()
        _contextUsageHistory.value = emptyList()
        _maxObservedTps.value = 1f
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

    fun toggleAllDatasets() {
        _chat.update { state ->
            val allIds = state.datasets.map { it.id }.toSet()
            val newSelected = if (state.selectedDatasetIds.size == allIds.size) {
                emptySet()
            } else {
                allIds
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

        _tpsHistory.value = emptyList()
        _contextUsageHistory.value = emptyList()
        _maxObservedTps.value = 1f

        val totalPrompts = selectedDatasets.sumOf { it.prompts.size }
        val allInputTokens = selectedDatasets.sumOf { ds ->
            ds.prompts.sumOf { estimateTokens(it.prompt) }
        }

        _chat.update {
            it.copy(
                currentStep = BenchmarkStep.EXECUTING,
                messages = emptyList(),
                totalPromptsInSelectedDatasets = totalPrompts,
                totalInputTokens = allInputTokens,
                processedInputTokens = 0,
                totalOutputTokens = 0,
                currentDatasetIndex = 0,
                currentPromptIndex = 0,
                benchmarkElapsedSeconds = 0L,
                benchmarkRemainingSeconds = null
            )
        }

        inferenceJob?.cancel()
        inferenceJob = viewModelScope.launch {
            try {
                val dbModelId = ModelService.registerOrGetModel(ctx, model.displayName) ?: 1L
                Log.d("LLM", "Benchmark model registered with ID: $dbModelId")

                // Fetch existing results to skip already processed prompts
                val existingResults = PromptService.fetchAllResults(ctx) ?: emptyList()
                val resultsMap = existingResults
                    .filter { it.idModel == dbModelId && it.idDevices == (deviceId ?: -1) }
                    .associateBy { it.idPrompt }
                val processedPromptIds = resultsMap.keys

                val overallStartTime = System.currentTimeMillis()
                var actualExecutedCount = 0
                val totalToRunInThisSession = selectedDatasets.sumOf { ds ->
                    ds.prompts.count { p -> !processedPromptIds.contains(p.id) }
                }

                // Background job for the timer
                launch {
                    while (isActive) {
                        delay(1000L)
                        val elapsed = (System.currentTimeMillis() - overallStartTime) / 1000
                        _chat.update { state ->
                            val remainingToRun = totalToRunInThisSession - actualExecutedCount

                            val remaining = if (actualExecutedCount > 0 && remainingToRun >= 0) {
                                val timePerPrompt = elapsed.toFloat() / actualExecutedCount
                                (timePerPrompt * remainingToRun).toLong()
                            } else null

                            state.copy(
                                benchmarkElapsedSeconds = elapsed,
                                benchmarkRemainingSeconds = remaining
                            )
                        }
                    }
                }

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

                var overallPromptCounter = 0
                for ((dIdx, dataset) in selectedDatasets.withIndex()) {
                    _chat.update { it.copy(currentDatasetIndex = dIdx, messages = emptyList()) }
                    chatHistory.clear()
                    _contextUsageHistory.value = emptyList()

                    for ((pIdx, promptItem) in dataset.prompts.withIndex()) {
                        _chat.update { it.copy(currentPromptIndex = overallPromptCounter) }

                        val isProcessed = processedPromptIds.contains(promptItem.id)
                        val promptInputTokens = estimateTokens(promptItem.prompt)

                        if (isProcessed) {
                            val previousResult = resultsMap[promptItem.id]
                            val previousOutputTokens = previousResult?.responseTokenCount ?: estimateTokens(previousResult?.response)

                            _chat.update { state ->
                                state.copy(
                                    processedInputTokens = state.processedInputTokens + promptInputTokens,
                                    totalOutputTokens = state.totalOutputTokens + previousOutputTokens
                                )
                            }

                            if (dataset.isConversational) {
                                // Rebuild history for conversational datasets
                                val previousResponse = resultsMap[promptItem.id]?.response ?: ""
                                chatHistory.add(mandiole.userTurn(promptItem.prompt))
                                chatHistory.add(mandiole.assistantTurn(previousResponse))

                                pruneHistory(model.contextSize)

                                _chat.update { state ->
                                    state.copy(
                                        messages = state.messages + ChatMessage(isUser = true, promptItem.prompt) +
                                                ChatMessage(isUser = false, previousResponse)
                                    )
                                }
                            }
                            Log.d("LLM", "Skipping prompt ${promptItem.id} (already benchmarked)")
                            overallPromptCounter++
                            continue
                        }

                        if (!dataset.isConversational) {
                            chatHistory.clear()
                            _contextUsageHistory.value = emptyList()
                            _chat.update { it.copy(messages = emptyList()) }
                        }

                        _tpsHistory.value = emptyList()
                        val promptText = promptItem.prompt
                        chatHistory.add(mandiole.userTurn(promptText))

                        // Prune history BEFORE generation to ensure the new prompt fits
                        pruneHistory(model.contextSize)

                        _chat.update { state ->
                            state.copy(
                                processedInputTokens = state.processedInputTokens + promptInputTokens,
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

                            // Calculate instantaneous TPS (including thinking tokens)
                            val elapsedMs = (currentTime - startTime).coerceAtLeast(1L)
                            val textTokens = estimateTokens(partial.text)
                            val thinkingTokens = estimateTokens(partial.thinkingText)
                            val totalTokens = textTokens + thinkingTokens

                            val currentTps = (totalTokens.toFloat() / (elapsedMs / 1000f))
                            if (currentTps > 0) {
                                val updatedHistory = _tpsHistory.value.toMutableList()
                                updatedHistory.add(currentTps)
                                if (updatedHistory.size > 60) updatedHistory.removeAt(0)
                                _tpsHistory.value = updatedHistory
                                if (currentTps > _maxObservedTps.value) {
                                    _maxObservedTps.value = currentTps
                                }
                            }

                            // Update context usage
                            val historyTokens = chatHistory.sumOf { turn ->
                                estimateTokens(turn.text) + estimateTokens(turn.thinkingText)
                            }
                            val currentUsage = (historyTokens + totalTokens).toFloat()
                            val updatedContextHistory = _contextUsageHistory.value.toMutableList()
                            updatedContextHistory.add(currentUsage)
                            if (updatedContextHistory.size > 60) updatedContextHistory.removeAt(0)
                            _contextUsageHistory.value = updatedContextHistory

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
                            state.copy(
                                messages = limitedMessages,
                                isGenerating = false,
                                totalOutputTokens = state.totalOutputTokens + tokenCount
                            )
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
                            actualExecutedCount++
                        } catch (e: Exception) {
                            Log.e("LLM", "Failed to submit result", e)
                        }
                        overallPromptCounter++
                    }
                }
                _chat.update { it.copy(currentPromptIndex = overallPromptCounter) }
            } catch (e: Exception) {
                Log.e("LLM", "Benchmark execution failed", e)
                _chat.update { state ->
                    state.copy(
                        isGenerating = false,
                        messages = state.messages + ChatMessage(
                            isUser = false,
                            text = "⚠️ Benchmark Error: ${e.localizedMessage ?: "Unknown error occurred"}"
                        )
                    )
                }
            }
        }
    }

    fun setThinkingEnabled(enabled: Boolean) {
        _chat.update { it.copy(thinkingEnabled = enabled) }
    }

    fun setRagEnabled(enabled: Boolean) {
        _chat.update { it.copy(ragEnabled = enabled) }
    }

    fun sendPrompt(prompt: String) {
        _tpsHistory.value = emptyList()
        Log.d("LLM", "--- sendPrompt pipeline started ---")
        Log.d("LLM", "User input: $prompt")
        Log.d("LLM", "RAG state: ${_chat.value.ragEnabled}")

        val originalHistory = chatHistory.toList()
        chatHistory.add(mandiole.userTurn(prompt))

        val model = currentModel
        if (model != null) {
            pruneHistory(model.contextSize)
        }

        _chat.update { state ->
            state.copy(
                isGenerating = true,
                messages = state.messages + ChatMessage(isUser = true, prompt) + ChatMessage(isUser = false, "")
            )
        }

        inferenceJob?.cancel()
        inferenceJob = viewModelScope.launch {
            try {
                var ragQuery: String? = null
                var ragResults: List<String>? = null
                var finalContextSynthesis: String? = null
                var ragAnalysisSteps = StringBuilder()

                if (_chat.value.ragEnabled) {
                    Log.d("LLM", "Phase 1: LLM 1 Audit")
                    
                    _chat.update { state ->
                        val newMessages = state.messages.toMutableList()
                        if (newMessages.isNotEmpty()) {
                            newMessages[newMessages.size - 1] = ChatMessage(
                                isUser = false,
                                text = "LLM 1: Analyzing intention...",
                                ragStatus = RagStatus.ANALYZING
                            )
                        }
                        state.copy(messages = newMessages)
                    }

                    val auditInstruction = """
                        SYSTEM: You are an Intention Auditor. 
                        Your job is to decide if the user's latest message requires information from external PDF documents.
                        - If the user asks about salary, contracts, dates, specific names, or data likely in their documents: Respond 'YES'.
                        - If it's a greeting, casual chat, or general knowledge: Respond 'NO'.
                        RESPOND ONLY WITH 'YES' OR 'NO'.
                        USER MESSAGE: $prompt
                    """.trimIndent()

                    val auditTurn = mandiole.userTurn(auditInstruction)
                    val auditResponse = mandiole.streamReply(listOf(auditTurn), thinkingEnabled = false) {}.text.trim()
                    Log.d("LLM", "LLM 1 Audit Response: $auditResponse")
                    ragAnalysisSteps.append("Intention Audit: $auditResponse\n")

                    if (auditResponse.contains("YES", ignoreCase = true)) {
                        Log.d("LLM", "Phase 2: LLM 1 Query Generation")
                        
                        _chat.update { state ->
                            val newMessages = state.messages.toMutableList()
                            if (newMessages.isNotEmpty()) {
                                newMessages[newMessages.size - 1] = newMessages.last().copy(
                                    text = "LLM 1: Formulating search query...",
                                    ragStatus = RagStatus.ANALYZING,
                                    ragReasoning = ragAnalysisSteps.toString()
                                )
                            }
                            state.copy(messages = newMessages)
                        }

                        val queryInstruction = "Generate a short search query to find the answer for '$prompt' in PDF documents. Output ONLY the search query."
                        val queryResponse = mandiole.streamReply(originalHistory.map { mandiole.userTurn(it.text) } + mandiole.userTurn(queryInstruction), thinkingEnabled = false) {}.text.trim()
                        ragQuery = queryResponse.removePrefix("\"").removeSuffix("\"")
                        Log.d("LLM", "LLM 1 Search Query: $ragQuery")
                        ragAnalysisSteps.append("Generated Query: $ragQuery\n")

                        _chat.update { state ->
                            val newMessages = state.messages.toMutableList()
                            if (newMessages.isNotEmpty()) {
                                newMessages[newMessages.size - 1] = newMessages.last().copy(
                                    text = "LLM 1: Searching documents for '$ragQuery'...",
                                    ragStatus = RagStatus.SEARCHING,
                                    ragQuery = ragQuery,
                                    ragReasoning = ragAnalysisSteps.toString()
                                )
                            }
                            state.copy(messages = newMessages)
                        }

                        // Search 1
                        var results = RAG.queryRAG(ragQuery).toList()
                        Log.d("LLM", "Search 1 found ${results.size} chunks")

                        // LLM 1: Validation
                        Log.d("LLM", "Phase 3: LLM 1 Validation")
                        val validationInstruction = "Does the following context contain the answer for '$prompt'?\n\nCONTEXT:\n${results.joinToString("\n")}\n\nRespond ONLY with 'YES' or 'NO'."
                        val validationResponse = mandiole.streamReply(listOf(mandiole.userTurn(validationInstruction)), thinkingEnabled = false) {}.text.trim()
                        Log.d("LLM", "LLM 1 Validation: $validationResponse")
                        ragAnalysisSteps.append("Context Validation: $validationResponse\n")

                        if (validationResponse.contains("NO", ignoreCase = true)) {
                            Log.d("LLM", "Phase 4: LLM 1 Retry with Query 2")
                            val query2Instruction = "The first search for '$ragQuery' failed. Generate a DIFFERENT query to find information for '$prompt'. Output ONLY the query."
                            val query2Response = mandiole.streamReply(listOf(mandiole.userTurn(query2Instruction)), thinkingEnabled = false) {}.text.trim()
                            val ragQuery2 = query2Response.removePrefix("\"").removeSuffix("\"")
                            Log.d("LLM", "LLM 1 Query 2: $ragQuery2")
                            ragAnalysisSteps.append("Retry Query: $ragQuery2\n")
                            
                            val results2 = RAG.queryRAG(ragQuery2).toList()
                            results = (results + results2).distinct()
                            Log.d("LLM", "Search 2 added ${results2.size} chunks")
                        }

                        // LLM 1: Synthesis
                        Log.d("LLM", "Phase 5: LLM 1 Synthesis")
                        _chat.update { state ->
                            val newMessages = state.messages.toMutableList()
                            if (newMessages.isNotEmpty()) {
                                newMessages[newMessages.size - 1] = newMessages.last().copy(
                                    text = "LLM 1: Extracting facts...",
                                    ragStatus = RagStatus.SYNTHESIZING,
                                    ragReasoning = ragAnalysisSteps.toString()
                                )
                            }
                            state.copy(messages = newMessages)
                        }

                        val synthesisInstruction = "Extract ONLY the specific facts from these chunks to answer '$prompt'. Be extremely brief. If not found, say 'NONE'.\n\nCHUNKS:\n${results.joinToString("\n")}"
                        val synthesis = mandiole.streamReply(listOf(mandiole.userTurn(synthesisInstruction)), thinkingEnabled = false) {}.text.trim()
                        
                        if (!synthesis.contains("NONE", ignoreCase = true)) {
                            finalContextSynthesis = synthesis
                            Log.d("LLM", "LLM 1 Synthesis: $finalContextSynthesis")
                            ragAnalysisSteps.append("Facts Extracted: $finalContextSynthesis\n")
                        } else {
                            Log.d("LLM", "LLM 1: Info not found in RAG.")
                            ragAnalysisSteps.append("Info not found in RAG.\n")
                        }
                        
                        ragResults = results
                        
                        _chat.update { state ->
                            val newMessages = state.messages.toMutableList()
                            if (newMessages.isNotEmpty()) {
                                newMessages[newMessages.size - 1] = newMessages.last().copy(
                                    ragStatus = RagStatus.SUCCESS,
                                    ragReasoning = ragAnalysisSteps.toString(),
                                    ragResults = ragResults
                                )
                            }
                            state.copy(messages = newMessages)
                        }
                    } else {
                        _chat.update { state ->
                            val newMessages = state.messages.toMutableList()
                            if (newMessages.isNotEmpty()) {
                                newMessages[newMessages.size - 1] = newMessages.last().copy(
                                    ragStatus = RagStatus.NOT_NEEDED,
                                    ragReasoning = "LLM 1 determined no search is needed."
                                )
                            }
                            state.copy(messages = newMessages)
                        }
                    }
                }

                // LLM 2: Final Response
                Log.d("LLM", "Phase 6: LLM 2 Final Output")
                val llm2History = originalHistory.toMutableList()
                val finalMessageWithContext = if (finalContextSynthesis != null) {
                    "FACTS FROM DOCUMENTS: $finalContextSynthesis\n\nUSER QUESTION: $prompt\n\nTask: Use the facts above to answer the user question naturally."
                } else {
                    prompt
                }
                llm2History.add(mandiole.userTurn(finalMessageWithContext))

                val isThinkingModeActive = _chat.value.thinkingEnabled
                val startTime = System.currentTimeMillis()

                mandiole.streamReply(
                    llm2History,
                    thinkingEnabled = isThinkingModeActive
                ) { response ->
                    val currentTime = System.currentTimeMillis()
                    
                    // Stats Calculation
                    val elapsedMs = (currentTime - startTime).coerceAtLeast(1L)
                    val textTokens = response.text.split(Regex("\\s+")).filter { it.isNotBlank() }.size
                    val thinkingTokens = response.thinkingText?.split(Regex("\\s+"))?.filter { it.isNotBlank() }?.size ?: 0
                    val totalTokens = textTokens + thinkingTokens
                    val currentTps = (totalTokens.toFloat() / (elapsedMs / 1000f))

                    if (currentTps > 0) {
                        val updatedHistory = _tpsHistory.value.toMutableList()
                        updatedHistory.add(currentTps)
                        if (updatedHistory.size > 60) updatedHistory.removeAt(0)
                        _tpsHistory.value = updatedHistory
                        if (currentTps > _maxObservedTps.value) _maxObservedTps.value = currentTps
                    }

                    val historyTokens = llm2History.sumOf { turn ->
                        turn.text.split(Regex("\\s+")).filter { it.isNotBlank() }.size
                    }
                    val updatedContextHistory = _contextUsageHistory.value.toMutableList()
                    updatedContextHistory.add((historyTokens + totalTokens).toFloat())
                    if (updatedContextHistory.size > 60) updatedContextHistory.removeAt(0)
                    _contextUsageHistory.value = updatedContextHistory

                    _chat.update { state ->
                        val newMessages = state.messages.toMutableList()
                        if (newMessages.isNotEmpty()) {
                            newMessages[newMessages.size - 1] = newMessages.last().copy(
                                text = response.text,
                                thinkingText = response.thinkingText,
                                thinkingDurationSeconds = if (response.thinkingText != null) ((currentTime - startTime)/1000).toInt() else null,
                                ragQuery = ragQuery,
                                ragResults = ragResults
                            )
                        }
                        state.copy(messages = newMessages)
                    }
                }.also { finalResponse ->
                    Log.d("LLM", "LLM 2 Response completed")
                    chatHistory.add(mandiole.assistantTurn(finalResponse.text))
                    _chat.update { state ->
                        val newMessages = state.messages.toMutableList()
                        if (newMessages.isNotEmpty()) {
                            newMessages[newMessages.size - 1] = newMessages.last().copy(
                                text = finalResponse.text,
                                thinkingText = finalResponse.thinkingText,
                                thinkingDurationSeconds = if (finalResponse.thinkingText != null) ((System.currentTimeMillis() - startTime)/1000).toInt() else null,
                                ragQuery = ragQuery,
                                ragResults = ragResults
                            )
                        }
                        state.copy(messages = newMessages, isGenerating = false)
                    }
                }
            } catch (e: Exception) {
                Log.e("LLM", "Multi-LLM Pipeline Error", e)
                _chat.update { state ->
                    state.copy(
                        isGenerating = false,
                        messages = state.messages + ChatMessage(isUser = false, text = "⚠️ Pipeline Error: ${e.localizedMessage}")
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

    private fun estimateTokens(text: String?): Int {
        if (text.isNullOrBlank()) return 0
        // Heuristic: ~3.5 chars per token for safety
        return (text.length / 3.5).toInt().coerceAtLeast(1)
    }

    private fun pruneHistory(maxTokens: Int) {
        // We want to keep at least 20% of the buffer for the model's new response.
        val safetyThreshold = (maxTokens * 0.8).toInt()

        var currentTotal = chatHistory.sumOf { turn ->
            estimateTokens(turn.text) + estimateTokens(turn.thinkingText)
        }

        // Remove oldest turns until it fits, but keep at least the current prompt (last item)
        while (chatHistory.size > 1 && currentTotal > safetyThreshold) {
            val removed = chatHistory.removeAt(0)
            currentTotal -= (estimateTokens(removed.text) + estimateTokens(removed.thinkingText))
            Log.d("LLM", "Pruning oldest context turn to stay under $safetyThreshold tokens")
        }
    }

    override fun onCleared() {
        super.onCleared()
        mandiole.close()
    }
}
