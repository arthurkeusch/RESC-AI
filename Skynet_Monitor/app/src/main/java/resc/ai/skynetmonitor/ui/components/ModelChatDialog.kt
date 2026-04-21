package resc.ai.skynetmonitor.ui.components

import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.Circle
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import fr.arthur.keusch.mandiole.model.ChatRole
import resc.ai.skynetmonitor.viewmodel.ChatMessage
import resc.ai.skynetmonitor.viewmodel.DeviceInfoViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelChatDialog(
    viewModel: DeviceInfoViewModel,
    onClose: () -> Unit
) {
    val chat by viewModel.benchmarkState.collectAsState()
    val downloadState by viewModel.downloadState.collectAsState()
    var userInput by remember { mutableStateOf("") }
    val listState = rememberLazyListState()

    LaunchedEffect(chat.messages.size, if (chat.messages.isNotEmpty()) chat.messages.last().text else "") {
        if (chat.messages.isNotEmpty()) {
            listState.animateScrollToItem(chat.messages.size - 1)
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
                // Header
                HeaderSection(chat.modelName, chat.isModelLoaded, downloadState)

                // Chat Messages
                LazyColumn(
                    state = listState,
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth(),
                    contentPadding = PaddingValues(16.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    items(chat.messages) { message ->
                        ChatBubble(message)
                    }
                }

                HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)

                // Input Area
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

                // Close Button
                TextButton(
                    onClick = { viewModel.stopBenchmark(); onClose() },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 8.dp)
                ) {
                    Text("Close Chat", color = MaterialTheme.colorScheme.outline)
                }
            }
        }
    }
}

@Composable
fun HeaderSection(modelName: String, isLoaded: Boolean, downloadState: resc.ai.skynetmonitor.service.DownloadState?) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f))
            .padding(16.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Column {
                Text(
                    text = modelName,
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold
                )
                if (downloadState != null) {
                    Text(
                        text = "Downloading... ${downloadState.progress}%",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.primary
                    )
                }
            }

            // Status Icon
            StatusIcon(isLoaded)
        }

        if (downloadState != null) {
            Spacer(Modifier.height(8.dp))
            LinearProgressIndicator(
                progress = { downloadState.progress / 100f },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(4.dp)
                    .clip(CircleShape),
            )
        } else if (!isLoaded) {
            Spacer(Modifier.height(8.dp))
            LinearProgressIndicator(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(2.dp)
                    .clip(CircleShape),
            )
        }
    }
}

@Composable
fun StatusIcon(isLoaded: Boolean) {
    val infiniteTransition = rememberInfiniteTransition(label = "status")
    val alpha by infiniteTransition.animateFloat(
        initialValue = 0.4f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(1000, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "alpha"
    )

    Row(verticalAlignment = Alignment.CenterVertically) {
        Icon(
            imageVector = Icons.Default.Circle,
            contentDescription = null,
            tint = if (isLoaded) Color(0xFF4CAF50) else Color(0xFFFFC107),
            modifier = Modifier
                .size(12.dp)
                .then(if (!isLoaded) Modifier.graphicsLayer { this.alpha = alpha } else Modifier)
        )
        Spacer(Modifier.width(6.dp))
        Text(
            text = if (isLoaded) "Ready" else "Loading",
            style = MaterialTheme.typography.labelMedium,
            color = if (isLoaded) Color(0xFF4CAF50) else Color(0xFFFFC107)
        )
    }
}

@Composable
fun ChatBubble(message: ChatMessage) {
    val isUser = message.role == ChatRole.USER
    val alignment = if (isUser) Alignment.End else Alignment.Start
    val containerColor = if (isUser) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.secondaryContainer
    val contentColor = if (isUser) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSecondaryContainer
    
    val shape = if (isUser) {
        RoundedCornerShape(20.dp, 20.dp, 4.dp, 20.dp)
    } else {
        RoundedCornerShape(20.dp, 20.dp, 20.dp, 4.dp)
    }

    Column(
        modifier = Modifier.fillMaxWidth(),
        horizontalAlignment = alignment
    ) {
        Surface(
            color = containerColor,
            contentColor = contentColor,
            shape = shape,
            tonalElevation = 2.dp,
            modifier = Modifier.widthIn(max = 300.dp)
        ) {
            Text(
                text = message.text,
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 12.dp),
                style = MaterialTheme.typography.bodyLarge.copy(lineHeight = 22.sp)
            )
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
        val buttonColor = if (isGenerating) MaterialTheme.colorScheme.error else if (buttonEnabled) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.surfaceVariant
        val iconColor = if (isGenerating) MaterialTheme.colorScheme.onError else if (buttonEnabled) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.4f)

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
