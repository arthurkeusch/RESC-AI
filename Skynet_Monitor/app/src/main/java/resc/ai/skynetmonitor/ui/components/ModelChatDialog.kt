package resc.ai.skynetmonitor.ui.components

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.collectIsDraggedAsState
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.Lightbulb
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import fr.arthur.keusch.mandiole.Mandiole
import resc.ai.skynetmonitor.viewmodel.BenchmarkStep
import resc.ai.skynetmonitor.viewmodel.ChatMessage
import resc.ai.skynetmonitor.viewmodel.DeviceInfoViewModel
import java.util.Locale

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelChatDialog(
    viewModel: DeviceInfoViewModel, onClose: () -> Unit
) {
    val chat by viewModel.benchmarkState.collectAsState()
    val localModels by viewModel.localModels.collectAsState()
    val downloadState by viewModel.downloadState.collectAsState()
    var userInput by remember { mutableStateOf("") }
    val listState = rememberLazyListState()
    var autoScrollEnabled by remember { mutableStateOf(true) }

    val configuration = androidx.compose.ui.platform.LocalConfiguration.current
    val isCompactScreen = configuration.screenWidthDp < 600

    val isDragged by listState.interactionSource.collectIsDraggedAsState()

    val currentView = LocalView.current
    DisposableEffect(Unit) {
        currentView.keepScreenOn = true
        onDispose {
            currentView.keepScreenOn = false
        }
    }

    LaunchedEffect(isDragged) {
        if (isDragged) {
            val layoutInfo = listState.layoutInfo
            val lastVisibleItem = layoutInfo.visibleItemsInfo.lastOrNull()
            val isAtBottom =
                lastVisibleItem != null && lastVisibleItem.index >= layoutInfo.totalItemsCount - 2
            if (!isAtBottom) autoScrollEnabled = false
        }
    }

    LaunchedEffect(
        chat.messages.size,
        if (chat.messages.isNotEmpty()) chat.messages.last().text else "",
        chat.isGenerating
    ) {
        if (chat.isGenerating && chat.messages.isNotEmpty() && chat.messages.last().text.isEmpty()) {
            autoScrollEnabled = true
        }
        if (autoScrollEnabled && chat.messages.isNotEmpty()) {
            listState.scrollToItem(chat.messages.size)
        }
    }

    Dialog(
        onDismissRequest = { viewModel.stopBenchmark(); onClose() },
        properties = DialogProperties(usePlatformDefaultWidth = false)
    ) {
        Surface(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            color = MaterialTheme.colorScheme.surface,
            shape = MaterialTheme.shapes.extraLarge,
            tonalElevation = 6.dp
        ) {
            Column(modifier = Modifier.fillMaxSize()) {
                when (chat.currentStep) {
                    BenchmarkStep.MODEL_SELECTION -> {
                        ModelSelectionStep(
                            localModels = localModels,
                            onModelSelected = { viewModel.selectModelForBenchmark(it) },
                            onClose = onClose
                        )
                    }

                    BenchmarkStep.DATASET_SELECTION -> {
                        DatasetSelectionStep(
                            datasets = chat.datasets,
                            selectedIds = chat.selectedDatasetIds,
                            thinkingEnabled = chat.thinkingEnabled,
                            onThinkingToggle = { viewModel.setThinkingEnabled(it) },
                            onToggleAll = { viewModel.toggleAllDatasets() },
                            onToggle = { viewModel.toggleDatasetSelection(it) },
                            onConfirm = { viewModel.runBenchmark() },
                            onBack = { viewModel.startBenchmarkFlow() })
                    }

                    BenchmarkStep.EXECUTING -> {
                        HeaderSection(
                            modelName = chat.modelName,
                            isLoaded = chat.isModelLoaded,
                            executionUnit = chat.executionUnit,
                            downloadState = downloadState,
                            canThink = chat.canThink,
                            thinkingEnabled = chat.thinkingEnabled,
                            onThinkingToggle = { viewModel.setThinkingEnabled(it) },
                            ragEnabled = chat.ragEnabled,
                            onRagToggle = { viewModel.setRagEnabled(it) },
                            isBenchmarking = chat.isBenchmarking,
                            showStatsPanel = chat.showStatsPanel,
                            onToggleStats = { viewModel.toggleStatsPanel() })

                        Row(
                            modifier = Modifier
                                .weight(1f)
                                .fillMaxWidth()
                        ) {
                            Column(modifier = Modifier.weight(if (chat.showStatsPanel) 0.7f else 1f)) {
                                LazyColumn(
                                    state = listState,
                                    modifier = Modifier
                                        .weight(1f)
                                        .fillMaxWidth(),
                                    contentPadding = PaddingValues(16.dp),
                                    verticalArrangement = Arrangement.spacedBy(12.dp)
                                ) {
                                    items(chat.messages) { message ->
                                        ChatBubble(message, chat.thinkingEnabled)
                                    }
                                    item {
                                        Spacer(
                                            modifier = Modifier
                                                .height(1.dp)
                                                .fillMaxWidth()
                                        )
                                    }
                                }
                            }

                            if (chat.showStatsPanel) {
                                VerticalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                                Column(
                                    modifier = Modifier
                                        .weight(0.35f)
                                        .fillMaxHeight()
                                        .padding(8.dp)
                                ) {
                                    Text(
                                        "Live Stats",
                                        style = MaterialTheme.typography.titleSmall,
                                        fontWeight = FontWeight.Bold,
                                        modifier = Modifier.padding(bottom = 8.dp)
                                    )
                                    val tpsHistory by viewModel.tpsHistory.collectAsState()
                                    val contextUsageHistory by viewModel.contextUsageHistory.collectAsState()
                                    val maxTps by viewModel.maxObservedTps.collectAsState()
                                    val currentModelDesc = localModels.find { it.displayName == chat.modelName }
                                    val maxContext = currentModelDesc?.contextSize ?: 1024

                                    LazyColumn(
                                        modifier = Modifier.weight(1f),
                                        verticalArrangement = Arrangement.spacedBy(8.dp)
                                    ) {
                                        val systemState = viewModel.systemState.value
                                        items(systemState.keys.toList()) { label ->
                                            val value = systemState[label] ?: ""
                                            val history =
                                                viewModel.historyData.value[label] ?: emptyList()
                                            val bounds = viewModel.getBoundsFor(label)
                                            val color = when {
                                                label.contains("RAM", ignoreCase = true) -> Color(
                                                    0xFF42A5F5
                                                )

                                                label.contains("Temp", ignoreCase = true) -> Color(
                                                    0xFFFFA726
                                                )

                                                label.contains(
                                                    "Battery", ignoreCase = true
                                                ) -> Color(0xFF9CCC65)

                                                else -> Color(0xFFBA68C8)
                                            }

                                            Card(
                                                modifier = Modifier.fillMaxWidth(),
                                                colors = CardDefaults.cardColors(
                                                    containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(
                                                        alpha = 0.3f
                                                    )
                                                )
                                            ) {
                                                Column(modifier = Modifier.padding(8.dp)) {
                                                    if (isCompactScreen) {
                                                        Column {
                                                            Text(
                                                                label,
                                                                style = MaterialTheme.typography.labelSmall,
                                                                color = MaterialTheme.colorScheme.primary
                                                            )
                                                            Text(
                                                                value,
                                                                style = MaterialTheme.typography.labelSmall,
                                                                fontWeight = FontWeight.Bold
                                                            )
                                                        }
                                                    } else {
                                                        Row(
                                                            modifier = Modifier.fillMaxWidth(),
                                                            horizontalArrangement = Arrangement.SpaceBetween
                                                        ) {
                                                            Text(
                                                                label,
                                                                style = MaterialTheme.typography.labelSmall,
                                                                color = MaterialTheme.colorScheme.primary
                                                            )
                                                            Text(
                                                                value,
                                                                style = MaterialTheme.typography.labelSmall,
                                                                fontWeight = FontWeight.Bold
                                                            )
                                                        }
                                                    }
                                                    if (history.isNotEmpty()) {
                                                        Spacer(Modifier.height(4.dp))
                                                        CompactMiniGraph(
                                                            data = history,
                                                            color = color,
                                                            minValue = bounds.first,
                                                            maxValue = bounds.second
                                                        )
                                                    }
                                                }
                                            }
                                        }

                                        item {
                                            Spacer(Modifier.height(if (isCompactScreen) 8.dp else 16.dp))
                                            Card(
                                                modifier = Modifier.fillMaxWidth(),
                                                colors = CardDefaults.cardColors(
                                                    containerColor = MaterialTheme.colorScheme.primaryContainer.copy(
                                                        alpha = 0.2f
                                                    )
                                                )
                                            ) {
                                                Column(modifier = Modifier.padding(if (isCompactScreen) 8.dp else 10.dp)) {
                                                    if (isCompactScreen) {
                                                        Column {
                                                            Text(
                                                                "Speed",
                                                                style = MaterialTheme.typography.labelSmall,
                                                                fontWeight = FontWeight.Bold,
                                                                color = MaterialTheme.colorScheme.primary
                                                            )
                                                            val currentTps = tpsHistory.lastOrNull() ?: 0f
                                                            Text(
                                                                String.format(
                                                                    Locale.getDefault(),
                                                                    "%.1f",
                                                                    currentTps
                                                                ),
                                                                style = MaterialTheme.typography.labelSmall,
                                                                fontWeight = FontWeight.Bold
                                                            )
                                                        }
                                                    } else {
                                                        Row(
                                                            modifier = Modifier.fillMaxWidth(),
                                                            horizontalArrangement = Arrangement.SpaceBetween
                                                        ) {
                                                            Text(
                                                                "Tokens/s Speed",
                                                                style = MaterialTheme.typography.labelSmall,
                                                                fontWeight = FontWeight.Bold,
                                                                color = MaterialTheme.colorScheme.primary
                                                            )
                                                            val currentTps = tpsHistory.lastOrNull() ?: 0f
                                                            Text(
                                                                String.format(
                                                                    Locale.getDefault(),
                                                                    "%.1f",
                                                                    currentTps
                                                                ),
                                                                style = MaterialTheme.typography.labelSmall,
                                                                fontWeight = FontWeight.Bold
                                                            )
                                                        }
                                                    }
                                                    Spacer(Modifier.height(6.dp))
                                                    CompactMiniGraph(
                                                        data = tpsHistory,
                                                        color = MaterialTheme.colorScheme.primary,
                                                        minValue = 0f,
                                                        maxValue = if (maxTps > 1f) maxTps else 1f
                                                    )
                                                    if (!isCompactScreen) {
                                                        Row(
                                                            modifier = Modifier.fillMaxWidth(),
                                                            horizontalArrangement = Arrangement.SpaceBetween
                                                        ) {
                                                            Text(
                                                                "0",
                                                                style = MaterialTheme.typography.labelSmall,
                                                                color = MaterialTheme.colorScheme.outline,
                                                                fontSize = 8.sp
                                                            )
                                                            Text(
                                                                String.format(
                                                                    Locale.getDefault(),
                                                                    "%.1f max",
                                                                    maxTps
                                                                ),
                                                                style = MaterialTheme.typography.labelSmall,
                                                                color = MaterialTheme.colorScheme.outline,
                                                                fontSize = 8.sp
                                                            )
                                                        }
                                                    }
                                                }
                                            }
                                        }

                                        item {
                                            Spacer(Modifier.height(if (isCompactScreen) 6.dp else 12.dp))
                                            Card(
                                                modifier = Modifier.fillMaxWidth(),
                                                colors = CardDefaults.cardColors(
                                                    containerColor = MaterialTheme.colorScheme.secondaryContainer.copy(
                                                        alpha = 0.2f
                                                    )
                                                )
                                            ) {
                                                Column(modifier = Modifier.padding(if (isCompactScreen) 8.dp else 10.dp)) {
                                                    if (isCompactScreen) {
                                                        Column {
                                                            Text(
                                                                "Context",
                                                                style = MaterialTheme.typography.labelSmall,
                                                                fontWeight = FontWeight.Bold,
                                                                color = MaterialTheme.colorScheme.secondary
                                                            )
                                                            val currentUsage =
                                                                contextUsageHistory.lastOrNull()?.toInt() ?: 0
                                                            Text(
                                                                "$currentUsage / $maxContext",
                                                                style = MaterialTheme.typography.labelSmall,
                                                                fontWeight = FontWeight.Bold
                                                            )
                                                        }
                                                    } else {
                                                        Row(
                                                            modifier = Modifier.fillMaxWidth(),
                                                            horizontalArrangement = Arrangement.SpaceBetween
                                                        ) {
                                                            Text(
                                                                "Context Usage",
                                                                style = MaterialTheme.typography.labelSmall,
                                                                fontWeight = FontWeight.Bold,
                                                                color = MaterialTheme.colorScheme.secondary
                                                            )
                                                            val currentUsage =
                                                                contextUsageHistory.lastOrNull()?.toInt() ?: 0
                                                            Text(
                                                                "$currentUsage / $maxContext",
                                                                style = MaterialTheme.typography.labelSmall,
                                                                fontWeight = FontWeight.Bold
                                                            )
                                                        }
                                                    }
                                                    Spacer(Modifier.height(6.dp))
                                                    CompactMiniGraph(
                                                        data = contextUsageHistory,
                                                        color = MaterialTheme.colorScheme.secondary,
                                                        minValue = 0f,
                                                        maxValue = maxContext.toFloat()
                                                    )
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)

                        if (chat.isBenchmarking) {
                            BenchmarkProgressSection(
                                currentDatasetIdx = chat.currentDatasetIndex,
                                totalDatasets = chat.selectedDatasetIds.size,
                                currentPromptIdx = chat.currentPromptIndex,
                                totalPrompts = chat.totalPromptsInSelectedDatasets,
                                elapsedSec = chat.benchmarkElapsedSeconds,
                                remainingSec = chat.benchmarkRemainingSeconds,
                                processedInputTokens = chat.processedInputTokens,
                                totalInputTokens = chat.totalInputTokens,
                                totalOutputTokens = chat.totalOutputTokens
                            )
                        } else {
                            InputSection(
                                userInput = userInput,
                                onValueChange = { userInput = it },
                                onSend = {
                                    if (userInput.isNotBlank()) {
                                        viewModel.sendPrompt(userInput); userInput = ""
                                    }
                                },
                                onStop = { viewModel.cancelGeneration() },
                                isGenerating = chat.isGenerating,
                                isModelLoaded = chat.isModelLoaded
                            )
                        }

                        TextButton(
                            onClick = { viewModel.stopBenchmark(); onClose() },
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(bottom = 8.dp)
                        ) {
                            Text(
                                text = if (chat.isBenchmarking) "Close Benchmark" else "Close Chat",
                                color = MaterialTheme.colorScheme.outline
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun BenchmarkProgressSection(
    currentDatasetIdx: Int,
    totalDatasets: Int,
    currentPromptIdx: Int,
    totalPrompts: Int,
    elapsedSec: Long,
    remainingSec: Long?,
    processedInputTokens: Int,
    totalInputTokens: Int,
    totalOutputTokens: Int
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        val overallProgress =
            if (totalPrompts > 0) currentPromptIdx.toFloat() / totalPrompts else 0f

        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Column {
                Text(
                    "Benchmark Progress",
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    "Elapsed: ${formatDuration(elapsedSec)}${remainingSec?.let { " • ETA: ${formatDuration(it)}" } ?: ""}",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.primary
                )
            }
            Text(
                "${currentPromptIdx}/${totalPrompts} Prompts",
                style = MaterialTheme.typography.labelSmall
            )
        }

        LinearProgressIndicator(
            progress = { overallProgress },
            modifier = Modifier
                .fillMaxWidth()
                .height(8.dp)
                .clip(CircleShape),
            color = MaterialTheme.colorScheme.primary,
            trackColor = MaterialTheme.colorScheme.surfaceVariant
        )

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(
                text = "Dataset ${currentDatasetIdx + 1} of ${totalDatasets}",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.outline
            )
            
            Text(
                text = "Tokens: IN $processedInputTokens/$totalInputTokens • OUT $totalOutputTokens",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.secondary,
                fontWeight = FontWeight.Medium
            )
        }
    }
}

private fun formatDuration(seconds: Long): String {
    val h = seconds / 3600
    val m = (seconds % 3600) / 60
    val s = seconds % 60
    return if (h > 0) {
        String.format(Locale.getDefault(), "%dh %02dm %02ds", h, m, s)
    } else {
        String.format(Locale.getDefault(), "%02dm %02ds", m, s)
    }
}

@Composable
fun ModelSelectionStep(
    localModels: List<Mandiole.ModelDescriptor>,
    onModelSelected: (Mandiole.ModelDescriptor) -> Unit,
    onClose: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        Text(
            "Step 1: Select a Local Model",
            style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.Bold
        )
        Spacer(Modifier.height(16.dp))
        if (localModels.isEmpty()) {
            Box(
                Modifier
                    .weight(1f)
                    .fillMaxWidth(), contentAlignment = Alignment.Center
            ) {
                Text(
                    "No local models found. Please download one first.",
                    color = MaterialTheme.colorScheme.error
                )
            }
        } else {
            LazyColumn(
                modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                items(localModels) { model ->
                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clickable { onModelSelected(model) },
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
                        )
                    ) {
                        Row(
                            modifier = Modifier.padding(16.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Icon(
                                Icons.Default.SmartToy,
                                contentDescription = null,
                                tint = MaterialTheme.colorScheme.primary
                            )
                            Spacer(Modifier.width(12.dp))
                            Column {
                                Text(model.displayName, fontWeight = FontWeight.Bold); Text(
                                model.sizeLabel, style = MaterialTheme.typography.bodySmall
                            )
                            }
                        }
                    }
                }
            }
        }
        TextButton(onClick = onClose, modifier = Modifier.fillMaxWidth()) { Text("Cancel") }
    }
}

@Composable
fun DatasetSelectionStep(
    datasets: List<resc.ai.skynetmonitor.service.DatasetItem>,
    selectedIds: Set<Int>,
    thinkingEnabled: Boolean,
    onThinkingToggle: (Boolean) -> Unit,
    onToggleAll: () -> Unit,
    onToggle: (Int) -> Unit,
    onConfirm: () -> Unit,
    onBack: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                "Step 2: Select Datasets",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold
            )
            
            TextButton(onClick = onToggleAll) {
                val allSelected = datasets.isNotEmpty() && selectedIds.size == datasets.size
                Text(if (allSelected) "Deselect All" else "Select All")
            }
        }
        Spacer(Modifier.height(16.dp))
        if (datasets.isNotEmpty()) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable { onThinkingToggle(!thinkingEnabled) }
                    .padding(vertical = 8.dp)) {
                Checkbox(checked = thinkingEnabled, onCheckedChange = onThinkingToggle)
                Spacer(Modifier.width(8.dp))
                Column {
                    Text(
                        "Enable Reasoning", fontWeight = FontWeight.Bold
                    ); Text(
                    "The model will show its thinking process (slower but more accurate)",
                    style = MaterialTheme.typography.bodySmall
                )
                }
            }
            Spacer(Modifier.height(8.dp))
        }
        LazyColumn(
            modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            items(datasets) { dataset ->
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable { onToggle(dataset.id) },
                    colors = CardDefaults.cardColors(
                        containerColor = if (selectedIds.contains(dataset.id)) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceVariant.copy(
                            alpha = 0.3f
                        )
                    )
                ) {
                    Row(
                        modifier = Modifier.padding(16.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Checkbox(
                            checked = selectedIds.contains(dataset.id),
                            onCheckedChange = { onToggle(dataset.id) })
                        Spacer(Modifier.width(8.dp))
                        Column {
                            Text(dataset.name, fontWeight = FontWeight.Bold)
                            dataset.description?.let {
                                Text(
                                    it, style = MaterialTheme.typography.bodySmall, maxLines = 2
                                )
                            }
                            if (dataset.isConversational) {
                                Badge(
                                    containerColor = MaterialTheme.colorScheme.secondary.copy(
                                        alpha = 0.2f
                                    )
                                ) {
                                    Text(
                                        "Conversational",
                                        style = MaterialTheme.typography.labelSmall
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
        Spacer(Modifier.height(16.dp))
        Row(
            modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            OutlinedButton(onClick = onBack, modifier = Modifier.weight(1f)) { Text("Back") }
            Button(
                onClick = onConfirm,
                modifier = Modifier.weight(1f),
                enabled = selectedIds.isNotEmpty()
            ) { Text("Start Execution") }
        }
    }
}

@Composable
fun HeaderSection(
    modelName: String,
    isLoaded: Boolean,
    executionUnit: String?,
    downloadState: resc.ai.skynetmonitor.service.DownloadState?,
    canThink: Boolean,
    thinkingEnabled: Boolean,
    onThinkingToggle: (Boolean) -> Unit,
    ragEnabled: Boolean,
    onRagToggle: (Boolean) -> Unit,
    isBenchmarking: Boolean,
    showStatsPanel: Boolean,
    onToggleStats: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f))
            .padding(horizontal = 16.dp, vertical = 12.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = modelName,
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.weight(1f, fill = false)
                    )
                    if (isLoaded) {
                        Spacer(Modifier.width(8.dp))
                        
                        val context = androidx.compose.ui.platform.LocalContext.current
                        // RAG Toggle
                        IconToggleButton(
                            checked = ragEnabled,
                            onCheckedChange = { enabled ->
                                if (enabled) {
                                    val indexedFiles = com.example.anhilyx.rescai.rag.RAG.getIndexedFiles()
                                    if (indexedFiles.isEmpty()) {
                                        android.widget.Toast.makeText(
                                            context,
                                            "⚠️ RAG index is empty. Please add PDF files first.",
                                            android.widget.Toast.LENGTH_LONG
                                        ).show()
                                        return@IconToggleButton
                                    }
                                }
                                onRagToggle(enabled)
                            },
                            modifier = Modifier.size(24.dp)
                        ) {
                            Icon(
                                imageVector = if (ragEnabled) Icons.Default.Dataset else Icons.Default.DatasetLinked,
                                contentDescription = "Toggle RAG",
                                tint = if (ragEnabled) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outline,
                                modifier = Modifier.size(18.dp)
                            )
                        }

                        if (canThink) {
                            Spacer(Modifier.width(8.dp))
                            IconToggleButton(
                                checked = thinkingEnabled,
                                onCheckedChange = onThinkingToggle,
                                modifier = Modifier.size(24.dp)
                            ) {
                                Icon(
                                    imageVector = if (thinkingEnabled) Icons.Default.Lightbulb else Icons.Outlined.Lightbulb,
                                    contentDescription = "Toggle Thinking",
                                    tint = if (thinkingEnabled) Color(0xFFFFC107) else MaterialTheme.colorScheme.outline,
                                    modifier = Modifier.size(18.dp)
                                )
                            }
                        }
                    }
                }
                if (downloadState != null) {
                    Text(
                        text = "Downloading... ${downloadState.progress}%",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.primary
                    )
                } else if (isLoaded) {
                    val statusText = buildString {
                        if (ragEnabled) append("RAG active")
                        if (canThink && thinkingEnabled) {
                            if (isNotEmpty()) append(" • ")
                            append("Reasoning active")
                        }
                    }
                    if (statusText.isNotEmpty()) {
                        Text(
                            text = statusText,
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.primary.copy(alpha = 0.8f)
                        )
                    }
                }
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                IconButton(onClick = onToggleStats) {
                    Icon(
                        imageVector = if (showStatsPanel) Icons.Default.BarChart else Icons.Default.Timeline,
                        contentDescription = "Toggle Stats",
                        tint = if (showStatsPanel) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outline
                    )
                }
                StatusIcon(isLoaded, executionUnit)
            }
        }
        if (downloadState != null) {
            Spacer(Modifier.height(8.dp))
            LinearProgressIndicator(
                progress = { downloadState.progress / 100f },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(4.dp)
                    .clip(CircleShape)
            )
        } else if (!isLoaded) {
            Spacer(Modifier.height(8.dp))
            LinearProgressIndicator(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(2.dp)
                    .clip(CircleShape)
            )
        }
    }
}

@Composable
fun StatusIcon(isLoaded: Boolean, executionUnit: String?) {
    val infiniteTransition = rememberInfiniteTransition(label = "status")
    val alpha by infiniteTransition.animateFloat(
        0.4f,
        1f,
        infiniteRepeatable(tween(1000, easing = LinearEasing), RepeatMode.Reverse),
        "alpha"
    )
    Column(horizontalAlignment = Alignment.End) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(
                imageVector = Icons.Default.Circle,
                contentDescription = null,
                tint = if (isLoaded) Color(0xFF4CAF50) else Color(0xFFFFC107),
                modifier = Modifier
                    .size(12.dp)
                    .then(if (!isLoaded) Modifier.graphicsLayer {
                        this.alpha = alpha
                    } else Modifier))
            Spacer(Modifier.width(6.dp))
            Text(
                text = if (isLoaded) "Ready" else "Loading",
                style = MaterialTheme.typography.labelMedium,
                color = if (isLoaded) Color(0xFF4CAF50) else Color(0xFFFFC107)
            )
        }
        if (isLoaded && executionUnit != null) {
            Spacer(Modifier.height(4.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                val isGpu = executionUnit.contains("GPU", ignoreCase = true)
                Icon(
                    imageVector = if (isGpu) Icons.Default.Bolt else Icons.Default.Memory,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.secondary,
                    modifier = Modifier.size(14.dp)
                )
                Spacer(Modifier.width(4.dp))
                Text(
                    text = executionUnit,
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.secondary
                )
            }
        }
    }
}

@Composable
fun ChatBubble(message: ChatMessage, isThinkingModeActive: Boolean) {
    val isUser = message.isUser
    val alignment = if (isUser) Alignment.End else Alignment.Start
    val containerColor =
        if (isUser) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.secondaryContainer
    val contentColor =
        if (isUser) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSecondaryContainer
    val shape = if (isUser) RoundedCornerShape(20.dp, 20.dp, 4.dp, 20.dp) else RoundedCornerShape(
        20.dp, 20.dp, 20.dp, 4.dp
    )
    Column(modifier = Modifier.fillMaxWidth(), horizontalAlignment = alignment) {
        Surface(
            color = containerColor,
            contentColor = contentColor,
            shape = shape,
            tonalElevation = 2.dp,
            modifier = Modifier.widthIn(max = 300.dp)
        ) {
            Column(modifier = Modifier.padding(horizontal = 16.dp, vertical = 12.dp)) {
                // Multi-LLM RAG Pipeline Display
                if (message.ragStatus != resc.ai.skynetmonitor.viewmodel.RagStatus.IDLE) {
                    var isRagExpanded by remember { mutableStateOf(false) }
                    val statusLabel = when(message.ragStatus) {
                        resc.ai.skynetmonitor.viewmodel.RagStatus.ANALYZING -> "LLM 1: Analyzing intention..."
                        resc.ai.skynetmonitor.viewmodel.RagStatus.SEARCHING -> "LLM 1: Searching documents..."
                        resc.ai.skynetmonitor.viewmodel.RagStatus.SYNTHESIZING -> "LLM 1: Synthesizing facts..."
                        resc.ai.skynetmonitor.viewmodel.RagStatus.SUCCESS -> "Used RAG context for this answer"
                        resc.ai.skynetmonitor.viewmodel.RagStatus.NOT_NEEDED -> "LLM 1: No search needed"
                        else -> ""
                    }
                    
                    Surface(
                        onClick = { isRagExpanded = !isRagExpanded },
                        color = MaterialTheme.colorScheme.primary.copy(alpha = 0.1f),
                        contentColor = contentColor,
                        shape = RoundedCornerShape(8.dp),
                        modifier = Modifier.padding(vertical = 4.dp).fillMaxWidth()
                    ) {
                        Column(modifier = Modifier.padding(8.dp)) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Icon(
                                    imageVector = if (message.ragStatus == resc.ai.skynetmonitor.viewmodel.RagStatus.NOT_NEEDED) Icons.Default.DatasetLinked else Icons.Default.Dataset,
                                    contentDescription = null,
                                    modifier = Modifier.size(16.dp),
                                    tint = if (message.ragStatus == resc.ai.skynetmonitor.viewmodel.RagStatus.SUCCESS) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outline
                                )
                                Spacer(Modifier.width(4.dp))
                                Text(
                                    text = statusLabel,
                                    style = MaterialTheme.typography.labelMedium,
                                    fontWeight = FontWeight.Bold
                                )
                                Spacer(Modifier.weight(1f))
                                Icon(
                                    imageVector = if (isRagExpanded) Icons.Default.ExpandLess else Icons.Default.ExpandMore,
                                    contentDescription = null,
                                    modifier = Modifier.size(16.dp)
                                )
                            }
                            if (isRagExpanded) {
                                Spacer(Modifier.height(4.dp))
                                if (message.ragReasoning != null) {
                                    Text(
                                        text = "LLM 1 Thinking Process:",
                                        style = MaterialTheme.typography.labelSmall,
                                        color = MaterialTheme.colorScheme.primary,
                                        fontWeight = FontWeight.Bold
                                    )
                                    Text(
                                        text = message.ragReasoning,
                                        style = MaterialTheme.typography.bodySmall,
                                        fontStyle = FontStyle.Italic,
                                        fontSize = 10.sp
                                    )
                                }
                                if (message.ragQuery != null) {
                                    HorizontalDivider(Modifier.padding(vertical = 4.dp), color = contentColor.copy(alpha = 0.2f))
                                    Text(
                                        "Search Query: ${message.ragQuery}",
                                        style = MaterialTheme.typography.labelSmall,
                                        fontWeight = FontWeight.SemiBold
                                    )
                                }
                                message.ragResults?.let { results ->
                                    HorizontalDivider(Modifier.padding(vertical = 4.dp), color = contentColor.copy(alpha = 0.2f))
                                    Text("Retrieved References (${results.size}):", style = MaterialTheme.typography.labelSmall)
                                    results.take(3).forEachIndexed { index, result ->
                                        Text(
                                            "Ref ${index + 1}: ${result.take(80)}...",
                                            style = MaterialTheme.typography.bodySmall,
                                            fontSize = 9.sp,
                                            lineHeight = 12.sp
                                        )
                                    }
                                }
                            }
                        }
                    }
                }

                if (message.thinkingText != null && message.thinkingText.isNotBlank()) {
                    var isExpanded by remember { mutableStateOf(false) }
                    Surface(
                        onClick = { isExpanded = !isExpanded },
                        color = Color.Transparent,
                        contentColor = contentColor.copy(alpha = 0.7f),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Column {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                modifier = Modifier.padding(vertical = 4.dp)
                            ) {
                                Icon(
                                    imageVector = if (isExpanded) Icons.Default.ExpandLess else Icons.Default.ExpandMore,
                                    contentDescription = null,
                                    modifier = Modifier.size(16.dp)
                                )
                                Spacer(Modifier.width(4.dp))
                                val durationText =
                                    message.thinkingDurationSeconds?.let { " ($it s)" } ?: ""
                                val label =
                                    if (!isThinkingModeActive) "Internal reasoning forced by model...$durationText" else if (isExpanded) "Thinking process$durationText" else "Model is thinking...$durationText"
                                Text(
                                    text = label,
                                    style = MaterialTheme.typography.labelMedium,
                                    fontStyle = FontStyle.Italic,
                                    color = if (!isThinkingModeActive) MaterialTheme.colorScheme.error.copy(
                                        alpha = 0.7f
                                    ) else contentColor.copy(alpha = 0.7f)
                                )
                            }
                            if (isExpanded) {
                                Text(
                                    text = message.thinkingText,
                                    style = MaterialTheme.typography.bodySmall.copy(
                                        fontStyle = FontStyle.Italic, lineHeight = 16.sp
                                    ),
                                    modifier = Modifier.padding(start = 20.dp, bottom = 8.dp)
                                )
                                HorizontalDivider(
                                    modifier = Modifier.padding(vertical = 8.dp),
                                    color = contentColor.copy(alpha = 0.2f)
                                )
                            }
                        }
                    }
                }
                if (message.text.isNotBlank()) {
                    MarkdownText(
                        text = message.text,
                        style = MaterialTheme.typography.bodyLarge.copy(lineHeight = 22.sp),
                        color = contentColor
                    )
                } else if (!isUser) {
                    TypingIndicator(contentColor)
                }
            }
        }
    }
}

@Composable
fun CompactMiniGraph(
    data: List<Float>, color: Color, minValue: Float, maxValue: Float
) {
    Canvas(
        modifier = Modifier
            .fillMaxWidth()
            .height(40.dp)
    ) {
        if (data.isEmpty()) return@Canvas

        val contentWidth = size.width
        val contentHeight = size.height

        val padded = if (data.size < 60) List(60 - data.size) { data.first() } + data else data
        val normalized = padded.map {
            ((it - minValue) / (maxValue - minValue).coerceAtLeast(0.0001f)).coerceIn(0f, 1f)
        }
        val stepX = contentWidth / (normalized.size - 1).coerceAtLeast(1)

        repeat(3) {
            val y = it * (contentHeight / 2)
            drawLine(
                color = Color.Gray.copy(alpha = 0.1f),
                start = Offset(0f, y),
                end = Offset(contentWidth, y)
            )
        }

        val path = Path()
        normalized.forEachIndexed { i, v ->
            val x = i * stepX
            val y = contentHeight - (v * contentHeight)
            if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
        }
        drawPath(path = path, color = color, style = Stroke(width = 2f))
    }
}

@Composable
fun TypingIndicator(color: Color) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        val infiniteTransition = rememberInfiniteTransition(label = "typing")
        val dotAlpha1 by infiniteTransition.animateFloat(
            0.2f, 1f, infiniteRepeatable(tween(600, 0), RepeatMode.Reverse), ""
        )
        val dotAlpha2 by infiniteTransition.animateFloat(
            0.2f, 1f, infiniteRepeatable(tween(600, 200), RepeatMode.Reverse), ""
        )
        val dotAlpha3 by infiniteTransition.animateFloat(
            0.2f, 1f, infiniteRepeatable(tween(600, 400), RepeatMode.Reverse), ""
        )
        listOf(dotAlpha1, dotAlpha2, dotAlpha3).forEach {
            Box(Modifier
                .padding(horizontal = 2.dp)
                .size(6.dp)
                .graphicsLayer { alpha = it }
                .background(color, CircleShape))
        }
    }
}

@Composable
fun MarkdownText(text: String, style: TextStyle, color: Color, modifier: Modifier = Modifier) {
    val parts = text.split("```")
    Column(modifier = modifier) {
        parts.forEachIndexed { index, part ->
            if (index % 2 == 1) {
                Surface(
                    color = color.copy(alpha = 0.05f),
                    shape = RoundedCornerShape(8.dp),
                    modifier = Modifier
                        .padding(vertical = 4.dp)
                        .fillMaxWidth()
                ) {
                    Text(
                        text = part.trim(), style = style.copy(
                            fontFamily = FontFamily.Monospace,
                            fontSize = (style.fontSize.value - 2).sp,
                            color = color.copy(alpha = 0.8f)
                        ), modifier = Modifier.padding(12.dp)
                    )
                }
            } else if (part.isNotBlank() || parts.size == 1) {
                val annotatedString = buildAnnotatedString {
                    var cursor = 0
                    val pattern = Regex("""(\*\*|__)(.*?)\1|(\*|_)(.*?)\3|(`)(.*?)\5""")
                    val matches = pattern.findAll(part)
                    matches.forEach { match ->
                        append(part.substring(cursor, match.range.first))
                        val bold = match.groups[1];
                        val italic = match.groups[3];
                        val code = match.groups[5]
                        when {
                            bold != null -> withStyle(SpanStyle(fontWeight = FontWeight.Bold)) {
                                append(
                                    match.groups[2]?.value ?: ""
                                )
                            }

                            italic != null -> withStyle(SpanStyle(fontStyle = FontStyle.Italic)) {
                                append(
                                    match.groups[4]?.value ?: ""
                                )
                            }

                            code != null -> withStyle(
                                SpanStyle(
                                    fontFamily = FontFamily.Monospace,
                                    background = color.copy(alpha = 0.1f)
                                )
                            ) { append(match.groups[6]?.value ?: "") }
                        }
                        cursor = match.range.last + 1
                    }
                    if (cursor < part.length) append(part.substring(cursor))
                }
                Text(text = annotatedString, style = style, color = color)
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun InputSection(
    userInput: String,
    onValueChange: (String) -> Unit,
    onSend: () -> Unit,
    onStop: () -> Unit,
    isGenerating: Boolean,
    isModelLoaded: Boolean
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(12.dp),
        verticalAlignment = Alignment.Bottom,
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        OutlinedTextField(
            value = userInput,
            onValueChange = onValueChange,
            modifier = Modifier.weight(1f),
            placeholder = { Text("Type a message...") },
            shape = RoundedCornerShape(24.dp),
            colors = OutlinedTextFieldDefaults.colors(
                focusedBorderColor = MaterialTheme.colorScheme.primary,
                unfocusedBorderColor = MaterialTheme.colorScheme.outline.copy(alpha = 0.3f)
            ),
            maxLines = 5,
            enabled = !isGenerating && isModelLoaded
        )
        val buttonEnabled = if (isGenerating) true else (userInput.isNotBlank() && isModelLoaded)
        val buttonColor =
            if (isGenerating) MaterialTheme.colorScheme.error else if (buttonEnabled) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.surfaceVariant
        val iconColor =
            if (isGenerating) MaterialTheme.colorScheme.onError else if (buttonEnabled) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSurfaceVariant.copy(
                alpha = 0.4f
            )
        IconButton(
            onClick = if (isGenerating) onStop else onSend,
            modifier = Modifier
                .size(48.dp)
                .clip(CircleShape)
                .background(buttonColor),
            enabled = buttonEnabled
        ) {
            Icon(
                imageVector = if (isGenerating) Icons.Default.Stop else Icons.AutoMirrored.Filled.Send,
                contentDescription = if (isGenerating) "Stop" else "Send",
                tint = iconColor
            )
        }
    }
}
