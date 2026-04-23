package resc.ai.skynetmonitor.ui.components

import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.interaction.collectIsDraggedAsState
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
import androidx.compose.ui.graphics.Color
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
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import fr.arthur.keusch.mandiole.model.ChatRole
import fr.arthur.keusch.mandiole.model.ModelDescriptor
import resc.ai.skynetmonitor.viewmodel.BenchmarkStep
import resc.ai.skynetmonitor.viewmodel.ChatMessage
import resc.ai.skynetmonitor.viewmodel.DeviceInfoViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelChatDialog(
    viewModel: DeviceInfoViewModel,
    onClose: () -> Unit
) {
    val chat by viewModel.benchmarkState.collectAsState()
    val localModels by viewModel.localModels.collectAsState()
    val downloadState by viewModel.downloadState.collectAsState()
    var userInput by remember { mutableStateOf("") }
    val listState = rememberLazyListState()
    var autoScrollEnabled by remember { mutableStateOf(true) }
    
    // Check if the list is being dragged by the user
    val isDragged by listState.interactionSource.collectIsDraggedAsState()

    // If the user drags the list away from the bottom, disable auto-scroll
    LaunchedEffect(isDragged) {
        if (isDragged) {
            val layoutInfo = listState.layoutInfo
            val lastVisibleItem = layoutInfo.visibleItemsInfo.lastOrNull()
            val isAtBottom = lastVisibleItem != null && 
                           lastVisibleItem.index >= layoutInfo.totalItemsCount - 2
            
            if (!isAtBottom) {
                autoScrollEnabled = false
            }
        }
    }

    // Main auto-scroll effect
    LaunchedEffect(chat.messages.size, if (chat.messages.isNotEmpty()) chat.messages.last().text else "", chat.isGenerating) {
        if (chat.isGenerating && chat.messages.isNotEmpty() && chat.messages.last().text.isEmpty()) {
            // Reset auto-scroll when a new turn starts
            autoScrollEnabled = true
        }
        
        if (autoScrollEnabled && chat.messages.isNotEmpty()) {
            // Use instant scroll for streaming to avoid animation overlap
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
                            onToggle = { viewModel.toggleDatasetSelection(it) },
                            onConfirm = { viewModel.runBenchmark() },
                            onBack = { viewModel.startBenchmarkFlow() }
                        )
                    }
                    BenchmarkStep.EXECUTING -> {
                        HeaderSection(
                            modelName = chat.modelName,
                            isLoaded = chat.isModelLoaded,
                            executionUnit = chat.executionUnit,
                            downloadState = downloadState,
                            canThink = chat.canThink,
                            thinkingEnabled = chat.thinkingEnabled,
                            onThinkingToggle = { viewModel.setThinkingEnabled(it) }
                        )

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
                            // Invisible anchor for auto-scroll to focus on the BOTTOM of the message
                            item {
                                Spacer(modifier = Modifier.height(1.dp).fillMaxWidth())
                            }
                        }

                        HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)

                        InputSection(
                            userInput = userInput,
                            onValueChange = { userInput = it },
                            onSend = {
                                if (userInput.isNotBlank()) {
                                    viewModel.sendPrompt(userInput)
                                    userInput = ""
                                }
                            },
                            onStop = { viewModel.cancelGeneration() },
                            isGenerating = chat.isGenerating,
                            isModelLoaded = chat.isModelLoaded
                        )

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
fun ModelSelectionStep(
    localModels: List<ModelDescriptor>,
    onModelSelected: (ModelDescriptor) -> Unit,
    onClose: () -> Unit
) {
    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
        Text("Step 1: Select a Local Model", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
        Spacer(Modifier.height(16.dp))
        
        if (localModels.isEmpty()) {
            Box(Modifier.weight(1f).fillMaxWidth(), contentAlignment = Alignment.Center) {
                Text("No local models found. Please download one first.", color = MaterialTheme.colorScheme.error)
            }
        } else {
            LazyColumn(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                items(localModels) { model ->
                    Card(
                        modifier = Modifier.fillMaxWidth().clickable { onModelSelected(model) },
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f))
                    ) {
                        Row(modifier = Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
                            Icon(Icons.Default.SmartToy, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                            Spacer(Modifier.width(12.dp))
                            Column {
                                Text(model.displayName, fontWeight = FontWeight.Bold)
                                Text(model.sizeLabel, style = MaterialTheme.typography.bodySmall)
                            }
                        }
                    }
                }
            }
        }
        
        TextButton(onClick = onClose, modifier = Modifier.fillMaxWidth()) {
            Text("Cancel")
        }
    }
}

@Composable
fun DatasetSelectionStep(
    datasets: List<resc.ai.skynetmonitor.service.DatasetItem>,
    selectedIds: Set<Int>,
    thinkingEnabled: Boolean,
    onThinkingToggle: (Boolean) -> Unit,
    onToggle: (Int) -> Unit,
    onConfirm: () -> Unit,
    onBack: () -> Unit
) {
    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
        Text("Step 2: Select Datasets", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
        Spacer(Modifier.height(16.dp))
        
        if (datasets.isNotEmpty()) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable { onThinkingToggle(!thinkingEnabled) }
                    .padding(vertical = 8.dp)
            ) {
                Checkbox(
                    checked = thinkingEnabled,
                    onCheckedChange = onThinkingToggle
                )
                Spacer(Modifier.width(8.dp))
                Column {
                    Text("Enable Reasoning", fontWeight = FontWeight.Bold)
                    Text("The model will show its thinking process (slower but more accurate)", style = MaterialTheme.typography.bodySmall)
                }
            }
            Spacer(Modifier.height(8.dp))
        }

        LazyColumn(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            items(datasets) { dataset ->
                Card(
                    modifier = Modifier.fillMaxWidth().clickable { onToggle(dataset.id) },
                    colors = CardDefaults.cardColors(
                        containerColor = if (selectedIds.contains(dataset.id)) 
                            MaterialTheme.colorScheme.primaryContainer 
                        else MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.3f)
                    )
                ) {
                    Row(modifier = Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
                        Checkbox(
                            checked = selectedIds.contains(dataset.id),
                            onCheckedChange = { onToggle(dataset.id) }
                        )
                        Spacer(Modifier.width(8.dp))
                        Column {
                            Text(dataset.name, fontWeight = FontWeight.Bold)
                            dataset.description?.let { 
                                Text(it, style = MaterialTheme.typography.bodySmall, maxLines = 2) 
                            }
                            if (dataset.isConversational) {
                                Badge(containerColor = MaterialTheme.colorScheme.secondary.copy(alpha = 0.2f)) {
                                    Text("Conversational", style = MaterialTheme.typography.labelSmall)
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Spacer(Modifier.height(16.dp))
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(onClick = onBack, modifier = Modifier.weight(1f)) {
                Text("Back")
            }
            Button(
                onClick = onConfirm, 
                modifier = Modifier.weight(1f),
                enabled = selectedIds.isNotEmpty()
            ) {
                Text("Start Execution")
            }
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
    onThinkingToggle: (Boolean) -> Unit
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
                    
                    if (canThink && isLoaded) {
                        Spacer(Modifier.width(8.dp))
                        IconToggleButton(
                            checked = thinkingEnabled,
                            onCheckedChange = onThinkingToggle,
                            modifier = Modifier.size(24.dp)
                        ) {
                            Icon(
                                imageVector = if (thinkingEnabled) Icons.Default.Lightbulb else Icons.Outlined.Lightbulb,
                                contentDescription = "Toggle Thinking Mode",
                                tint = if (thinkingEnabled) Color(0xFFFFC107) else MaterialTheme.colorScheme.outline,
                                modifier = Modifier.size(18.dp)
                            )
                        }
                    }
                }
                
                if (downloadState != null) {
                    Text(
                        text = "Downloading... ${downloadState.progress}%",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.primary
                    )
                } else if (canThink && isLoaded) {
                    Text(
                        text = if (thinkingEnabled) "Reasoning active" else "Reasoning disabled",
                        style = MaterialTheme.typography.labelSmall,
                        color = if (thinkingEnabled) Color(0xFFFFC107) else MaterialTheme.colorScheme.outline
                    )
                }
            }

            StatusIcon(isLoaded, executionUnit)
        }

        if (downloadState != null) {
            Spacer(Modifier.height(8.dp))
            LinearProgressIndicator(
                progress = { downloadState.progress / 100f },
                modifier = Modifier.fillMaxWidth().height(4.dp).clip(CircleShape),
            )
        } else if (!isLoaded) {
            Spacer(Modifier.height(8.dp))
            LinearProgressIndicator(
                modifier = Modifier.fillMaxWidth().height(2.dp).clip(CircleShape),
            )
        }
    }
}

@Composable
fun StatusIcon(isLoaded: Boolean, executionUnit: String?) {
    val infiniteTransition = rememberInfiniteTransition(label = "status")
    val alpha by infiniteTransition.animateFloat(
        initialValue = 0.4f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(tween(1000, easing = LinearEasing), repeatMode = RepeatMode.Reverse),
        label = "alpha"
    )

    Column(horizontalAlignment = Alignment.End) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(
                imageVector = Icons.Default.Circle,
                contentDescription = null,
                tint = if (isLoaded) Color(0xFF4CAF50) else Color(0xFFFFC107),
                modifier = Modifier.size(12.dp).then(if (!isLoaded) Modifier.graphicsLayer { this.alpha = alpha } else Modifier)
            )
            Spacer(Modifier.width(6.dp))
            Text(text = if (isLoaded) "Ready" else "Loading", style = MaterialTheme.typography.labelMedium, color = if (isLoaded) Color(0xFF4CAF50) else Color(0xFFFFC107))
        }
        
        if (isLoaded && executionUnit != null) {
            Spacer(Modifier.height(4.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                val isGpu = executionUnit.contains("GPU", ignoreCase = true)
                Icon(imageVector = if (isGpu) Icons.Default.Bolt else Icons.Default.Memory, contentDescription = null, tint = MaterialTheme.colorScheme.secondary, modifier = Modifier.size(14.dp))
                Spacer(Modifier.width(4.dp))
                Text(text = executionUnit, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.secondary)
            }
        }
    }
}

@Composable
fun ChatBubble(message: ChatMessage, isThinkingModeActive: Boolean) {
    val isUser = message.role == ChatRole.USER
    val alignment = if (isUser) Alignment.End else Alignment.Start
    val containerColor = if (isUser) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.secondaryContainer
    val contentColor = if (isUser) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSecondaryContainer
    val shape = if (isUser) RoundedCornerShape(20.dp, 20.dp, 4.dp, 20.dp) else RoundedCornerShape(20.dp, 20.dp, 20.dp, 4.dp)

    Column(modifier = Modifier.fillMaxWidth(), horizontalAlignment = alignment) {
        Surface(color = containerColor, contentColor = contentColor, shape = shape, tonalElevation = 2.dp, modifier = Modifier.widthIn(max = 300.dp)) {
            Column(modifier = Modifier.padding(horizontal = 16.dp, vertical = 12.dp)) {
                if (message.thinkingText != null && message.thinkingText.isNotBlank()) {
                    var isExpanded by remember { mutableStateOf(false) }
                    Surface(onClick = { isExpanded = !isExpanded }, color = Color.Transparent, contentColor = contentColor.copy(alpha = 0.7f), shape = RoundedCornerShape(8.dp)) {
                        Column {
                            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(vertical = 4.dp)) {
                                Icon(imageVector = if (isExpanded) Icons.Default.ExpandLess else Icons.Default.ExpandMore, contentDescription = null, modifier = Modifier.size(16.dp))
                                Spacer(Modifier.width(4.dp))
                                val durationText = message.thinkingDurationSeconds?.let { " ($it s)" } ?: ""
                                val label = if (!isThinkingModeActive) "Internal reasoning forced by model...$durationText" else if (isExpanded) "Thinking process$durationText" else "Model is thinking...$durationText"
                                Text(text = label, style = MaterialTheme.typography.labelMedium, fontStyle = FontStyle.Italic, color = if (!isThinkingModeActive) MaterialTheme.colorScheme.error.copy(alpha = 0.7f) else contentColor.copy(alpha = 0.7f))
                            }
                            if (isExpanded) {
                                Text(text = message.thinkingText, style = MaterialTheme.typography.bodySmall.copy(fontStyle = FontStyle.Italic, lineHeight = 16.sp), modifier = Modifier.padding(start = 20.dp, bottom = 8.dp))
                                HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp), color = contentColor.copy(alpha = 0.2f))
                            }
                        }
                    }
                }
                if (message.text.isNotBlank()) {
                    MarkdownText(text = message.text, style = MaterialTheme.typography.bodyLarge.copy(lineHeight = 22.sp), color = contentColor)
                } else if (!isUser) {
                    TypingIndicator(contentColor)
                }
            }
        }
    }
}

@Composable
fun TypingIndicator(color: Color) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        val infiniteTransition = rememberInfiniteTransition(label = "typing")
        val dotAlpha1 by infiniteTransition.animateFloat(0.2f, 1f, infiniteRepeatable(tween(600, 0), RepeatMode.Reverse), "")
        val dotAlpha2 by infiniteTransition.animateFloat(0.2f, 1f, infiniteRepeatable(tween(600, 200), RepeatMode.Reverse), "")
        val dotAlpha3 by infiniteTransition.animateFloat(0.2f, 1f, infiniteRepeatable(tween(600, 400), RepeatMode.Reverse), "")
        listOf(dotAlpha1, dotAlpha2, dotAlpha3).forEach { Box(Modifier.padding(horizontal = 2.dp).size(6.dp).graphicsLayer { alpha = it }.background(color, CircleShape)) }
    }
}

@Composable
fun MarkdownText(text: String, style: TextStyle, color: Color, modifier: Modifier = Modifier) {
    val parts = text.split("```")
    Column(modifier = modifier) {
        parts.forEachIndexed { index, part ->
            if (index % 2 == 1) {
                Surface(color = color.copy(alpha = 0.05f), shape = RoundedCornerShape(8.dp), modifier = Modifier.padding(vertical = 4.dp).fillMaxWidth()) {
                    Text(text = part.trim(), style = style.copy(fontFamily = FontFamily.Monospace, fontSize = (style.fontSize.value - 2).sp, color = color.copy(alpha = 0.8f)), modifier = Modifier.padding(12.dp))
                }
            } else if (part.isNotBlank() || parts.size == 1) {
                val annotatedString = buildAnnotatedString {
                    var cursor = 0
                    val pattern = Regex("""(\*\*|__)(.*?)\1|(\*|_)(.*?)\3|(`)(.*?)\5""")
                    val matches = pattern.findAll(part)
                    matches.forEach { match ->
                        append(part.substring(cursor, match.range.first))
                        val bold = match.groups[1]; val italic = match.groups[3]; val code = match.groups[5]
                        when {
                            bold != null -> withStyle(SpanStyle(fontWeight = FontWeight.Bold)) { append(match.groups[2]?.value ?: "") }
                            italic != null -> withStyle(SpanStyle(fontStyle = FontStyle.Italic)) { append(match.groups[4]?.value ?: "") }
                            code != null -> withStyle(SpanStyle(fontFamily = FontFamily.Monospace, background = color.copy(alpha = 0.1f))) { append(match.groups[6]?.value ?: "") }
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
fun InputSection(userInput: String, onValueChange: (String) -> Unit, onSend: () -> Unit, onStop: () -> Unit, isGenerating: Boolean, isModelLoaded: Boolean) {
    Row(modifier = Modifier.fillMaxWidth().padding(12.dp), verticalAlignment = Alignment.Bottom, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        OutlinedTextField(value = userInput, onValueChange = onValueChange, modifier = Modifier.weight(1f), placeholder = { Text("Type a message...") }, shape = RoundedCornerShape(24.dp), colors = OutlinedTextFieldDefaults.colors(focusedBorderColor = MaterialTheme.colorScheme.primary, unfocusedBorderColor = MaterialTheme.colorScheme.outline.copy(alpha = 0.3f)), maxLines = 5, enabled = !isGenerating && isModelLoaded)
        val buttonEnabled = if (isGenerating) true else (userInput.isNotBlank() && isModelLoaded)
        val buttonColor = if (isGenerating) MaterialTheme.colorScheme.error else if (buttonEnabled) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.surfaceVariant
        val iconColor = if (isGenerating) MaterialTheme.colorScheme.onError else if (buttonEnabled) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.4f)
        IconButton(onClick = if (isGenerating) onStop else onSend, modifier = Modifier.size(48.dp).clip(CircleShape).background(buttonColor), enabled = buttonEnabled) {
            Icon(imageVector = if (isGenerating) Icons.Default.Stop else Icons.AutoMirrored.Filled.Send, contentDescription = if (isGenerating) "Stop" else "Send", tint = iconColor)
        }
    }
}
