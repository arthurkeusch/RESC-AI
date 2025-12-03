package resc.ai.skynetmonitor.ui.components

import android.annotation.SuppressLint
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import resc.ai.skynetmonitor.service.DownloadState
import resc.ai.skynetmonitor.service.ModelService

data class ModelInfo(
    val name: String,
    val sizeBytes: Long,
    val parameters: String
)

@Composable
fun ModelSelectionDialog(
    models: List<ModelInfo>,
    selectedModel: ModelInfo?,
    downloadState: DownloadState?,
    onDismiss: () -> Unit,
    onConfirm: (ModelInfo) -> Unit
) {
    var selected by remember { mutableStateOf(selectedModel) }
    val isDownloading = downloadState != null

    Dialog(onDismissRequest = onDismiss) {
        Surface(
            shape = MaterialTheme.shapes.medium,
            color = MaterialTheme.colorScheme.surface,
            tonalElevation = 6.dp,
            modifier = Modifier
                .fillMaxWidth(0.95f)
                .wrapContentHeight()
        ) {
            if (!isDownloading) {
                Column(
                    modifier = Modifier
                        .padding(16.dp)
                        .fillMaxWidth(),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Text(
                        "Select a model",
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                    )
                    LazyColumn(
                        modifier = Modifier
                            .weight(1f, fill = false)
                            .fillMaxWidth()
                            .padding(bottom = 8.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        items(models) { model ->
                            val isSelected = model == selected
                            Surface(
                                shape = MaterialTheme.shapes.small,
                                tonalElevation = if (isSelected) 4.dp else 1.dp,
                                border = BorderStroke(
                                    if (isSelected) 2.dp else 1.dp,
                                    if (isSelected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outlineVariant
                                ),
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .clickable { selected = model }
                            ) {
                                Column(
                                    Modifier
                                        .padding(12.dp)
                                        .fillMaxWidth()
                                ) {
                                    Row(
                                        modifier = Modifier.fillMaxWidth(),
                                        horizontalArrangement = Arrangement.SpaceBetween,
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        Text(
                                            model.name,
                                            fontSize = 18.sp,
                                            fontWeight = FontWeight.SemiBold
                                        )
                                        Text(
                                            ModelService.formatSize(model.sizeBytes),
                                            fontSize = 14.sp,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant
                                        )
                                    }
                                    Spacer(Modifier.height(4.dp))
                                    Text(
                                        model.parameters,
                                        fontSize = 13.sp,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                }
                            }
                        }
                    }
                    Button(
                        onClick = { selected?.let { onConfirm(it) } },
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(top = 4.dp),
                        enabled = selected != null
                    ) {
                        Text("Select")
                    }
                }
            } else {
                val st = downloadState
                val progress = (st.progress / 100f).coerceIn(0f, 1f)
                Column(
                    modifier = Modifier
                        .padding(16.dp)
                        .fillMaxWidth(),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Text(
                        "Downloading ${st.name}",
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
                    )
                    LinearProgressIndicator(
                        progress = { progress },
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(8.dp)
                    )
                    val totalStr = ModelService.formatSize(st.totalBytes)
                    val recvStr = ModelService.formatSize(st.bytesReceived)
                    val speedStr = if (st.speedBytesPerSec > 0)
                        "${ModelService.formatSize(st.speedBytesPerSec)}/s"
                    else "—"
                    val etaStr = if (st.etaSeconds >= 0) formatEta(st.etaSeconds) else "—"
                    Text("$recvStr / $totalStr")
                    Text("Speed: $speedStr")
                    Text("ETA: $etaStr")
                    Spacer(Modifier.height(8.dp))
                }
            }
        }
    }
}

@SuppressLint("DefaultLocale")
private fun formatEta(sec: Long): String {
    val h = sec / 3600
    val m = (sec % 3600) / 60
    val s = sec % 60
    return when {
        h > 0 -> String.format("%dh %02dm %02ds", h, m, s)
        m > 0 -> String.format("%dm %02ds", m, s)
        else -> String.format("%ds", s)
    }
}
