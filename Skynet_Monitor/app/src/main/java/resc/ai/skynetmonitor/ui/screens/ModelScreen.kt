package resc.ai.skynetmonitor.ui.screens

import android.annotation.SuppressLint
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CloudDownload
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import resc.ai.skynetmonitor.service.DownloadState
import resc.ai.skynetmonitor.service.ModelService
import resc.ai.skynetmonitor.service.RemoteModel
import resc.ai.skynetmonitor.ui.components.ModelCard
import resc.ai.skynetmonitor.ui.theme.SkynetMonitorTheme
import resc.ai.skynetmonitor.viewmodel.DeviceInfoViewModel

@Composable
fun ModelScreen(innerPadding: PaddingValues, viewModel: DeviceInfoViewModel = viewModel()) {
    val models by viewModel.remoteModels.collectAsState()
    val downloadState by viewModel.downloadState.collectAsState()
    val isDeleting by viewModel.isDeleting.collectAsState()
    val lastDeleteCompleted by viewModel.lastDeleteCompleted.collectAsState()

    var actionModel by remember { mutableStateOf<RemoteModel?>(null) }
    var showMenu by remember { mutableStateOf(false) }
    var showDeleteConfirm by remember { mutableStateOf(false) }

    LaunchedEffect(Unit) {
        viewModel.loadModelsRemote()
    }

    LaunchedEffect(downloadState?.progress) {
        val st = downloadState
        if (st != null && st.progress >= 100) {
            viewModel.clearDownloadState()
        }
    }

    LaunchedEffect(lastDeleteCompleted) {
        val current = actionModel
        if (lastDeleteCompleted != null && current != null && current.filename == lastDeleteCompleted) {
            showDeleteConfirm = false
            showMenu = false
            actionModel = null
            viewModel.consumeDeleteEvent()
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .padding(innerPadding)
    ) {
        if (models.isEmpty()) {
            Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text("No models available")
            }
        } else {
            LazyVerticalGrid(
                columns = GridCells.Adaptive(minSize = 500.dp),
                modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(12.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                items(items = models, key = { it.filename }) { model ->
                    ModelCard(
                        model = model,
                        onClick = {
                            actionModel = it
                            showMenu = true
                        }
                    )
                }
            }
        }
    }

    val current = actionModel
    if (showMenu && current != null) {
        val local = current.isLocal
        AlertDialog(
            onDismissRequest = { showMenu = false },
            title = { Text(current.name) },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    if (local) {
                        Surface(
                            tonalElevation = 1.dp,
                            onClick = { showDeleteConfirm = true }
                        ) {
                            Row(
                                Modifier
                                    .fillMaxWidth()
                                    .padding(12.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Icon(Icons.Filled.Delete, contentDescription = null, tint = MaterialTheme.colorScheme.error)
                                Spacer(Modifier.width(12.dp))
                                Text("Delete model", color = MaterialTheme.colorScheme.error)
                            }
                        }
                    } else {
                        Surface(
                            tonalElevation = 1.dp,
                            onClick = {
                                showMenu = false
                                viewModel.downloadModel(current)
                            }
                        ) {
                            Row(
                                Modifier
                                    .fillMaxWidth()
                                    .padding(12.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Icon(Icons.Filled.CloudDownload, contentDescription = null)
                                Spacer(Modifier.width(12.dp))
                                Text("Download model")
                            }
                        }
                    }
                }
            },
            confirmButton = {},
            dismissButton = { TextButton(onClick = { showMenu = false }) { Text("Cancel") } }
        )
    }

    val modelToDelete = actionModel
    if (showDeleteConfirm && modelToDelete != null) {
        AlertDialog(
            onDismissRequest = { if (!isDeleting) showDeleteConfirm = false },
            title = { Text("Confirm deletion") },
            text = { Text("Are you sure you want to delete ${modelToDelete.name}?") },
            confirmButton = {
                TextButton(
                    onClick = { viewModel.deleteLocalModel(modelToDelete) },
                    enabled = !isDeleting
                ) { Text(if (isDeleting) "Deleting…" else "Delete", color = MaterialTheme.colorScheme.error) }
            },
            dismissButton = {
                TextButton(
                    onClick = { showDeleteConfirm = false },
                    enabled = !isDeleting
                ) { Text("Cancel") }
            }
        )
    }

    val st: DownloadState? = downloadState
    if (st != null && st.progress < 100) {
        val progress = (st.progress / 100f).coerceIn(0f, 1f)
        AlertDialog(
            onDismissRequest = {},
            title = { Text("Downloading ${st.name}") },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    LinearProgressIndicator(
                        progress = { progress },
                        modifier = Modifier.fillMaxWidth()
                    )
                    val totalStr = ModelService.formatSize(st.totalBytes)
                    val recvStr = ModelService.formatSize(st.bytesReceived)
                    val speedStr =
                        if (st.speedBytesPerSec > 0) "${ModelService.formatSize(st.speedBytesPerSec)}/s" else "—"
                    val etaStr = if (st.etaSeconds >= 0) formatEta(st.etaSeconds) else "—"
                    Text("$recvStr / $totalStr")
                    Text("Speed: $speedStr")
                    Text("ETA: $etaStr")
                }
            },
            confirmButton = {
                TextButton(onClick = { viewModel.cancelDownload() }) { Text("Close") }
            }
        )
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

@Preview(showBackground = true)
@Composable
fun ModelScreenPreview() {
    SkynetMonitorTheme {
        ModelScreen(PaddingValues())
    }
}
