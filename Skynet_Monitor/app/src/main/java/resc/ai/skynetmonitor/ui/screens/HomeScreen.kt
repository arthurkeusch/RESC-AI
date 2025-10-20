package resc.ai.skynetmonitor.ui.screens

import android.widget.EditText
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ExpandLess
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import resc.ai.skynetmonitor.service.ModelService
import resc.ai.skynetmonitor.ui.components.InfoCard
import resc.ai.skynetmonitor.ui.components.ModelChatDialog
import resc.ai.skynetmonitor.ui.components.ModelInfo
import resc.ai.skynetmonitor.ui.components.ModelSelectionDialog
import resc.ai.skynetmonitor.ui.theme.SkynetMonitorTheme
import resc.ai.skynetmonitor.viewmodel.DeviceInfoViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(innerPadding: PaddingValues, viewModel: DeviceInfoViewModel = viewModel()) {
    var showDialog by remember { mutableStateOf(false) }
    var selectedModel by remember { mutableStateOf<ModelInfo?>(null) }
    var hardwareExpanded by remember { mutableStateOf(false) }
    var systemExpanded by remember { mutableStateOf(true) }
    var apiUrl by remember { mutableStateOf(ModelService.API_BASE) }

    val models by viewModel.models.collectAsState()
    val downloadState by viewModel.downloadState.collectAsState()
    val chatState by viewModel.benchmarkState.collectAsState()

    LaunchedEffect(Unit) { viewModel.loadModels() }

    val hardwareInfo = viewModel.hardwareInfo.value
    val systemState = viewModel.systemState.value

    if (showDialog) {
        ModelSelectionDialog(
            models = models,
            selectedModel = selectedModel,
            downloadState = downloadState,
            onDismiss = { showDialog = false },
            onConfirm = { model ->
                viewModel.selectModel(model) { done ->
                    selectedModel = done
                    showDialog = false
                }
            }
        )
    }

    if (chatState.isRunning) {
        ModelChatDialog(
            viewModel = viewModel,
            onClose = { viewModel.stopBenchmark() }
        )
    }

    Scaffold(
        modifier = Modifier.padding(innerPadding),
        topBar = {
            Surface(
                tonalElevation = 3.dp,
                shadowElevation = 4.dp,
                color = MaterialTheme.colorScheme.surface
            ) {
                Column {
                    // Row to select the API url
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 16.dp, vertical = 12.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        TextField(
                            value = apiUrl,
                            onValueChange = { newValue ->
                                apiUrl = newValue
                                viewModel.setApiUrl(newValue) // ici on envoie le contenu actuel
                            },
                            modifier = Modifier.weight(1f),
                            label = { Text("API:") },
                        )
                    }

                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 16.dp, vertical = 12.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        OutlinedButton(
                            onClick = { showDialog = true },
                            enabled = downloadState == null && !chatState.isRunning,
                            modifier = Modifier
                                .widthIn(min = 160.dp)
                                .height(48.dp)
                        ) {
                            Text(selectedModel?.name ?: "Select model", fontSize = 15.sp)
                        }
                        Button(
                            onClick = { selectedModel?.let { viewModel.startBenchmarkFor(it) } },
                            enabled = selectedModel != null && downloadState == null && !chatState.isRunning,
                            modifier = Modifier
                                .widthIn(min = 160.dp)
                                .height(48.dp)
                        ) {
                            Text("Start Benchmark", fontSize = 15.sp)
                        }
                    }
                }
            }
        }
    ) { padding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 16.dp)
                .padding(padding),
            verticalArrangement = Arrangement.spacedBy(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            item {
                Column(modifier = Modifier.fillMaxWidth()) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clickable { hardwareExpanded = !hardwareExpanded }
                            .padding(vertical = 8.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text("Hardware Information")
                        Icon(
                            imageVector = if (hardwareExpanded) Icons.Default.ExpandLess else Icons.Default.ExpandMore,
                            contentDescription = null
                        )
                    }
                    AnimatedVisibility(visible = hardwareExpanded) {
                        Column(
                            verticalArrangement = Arrangement.spacedBy(8.dp),
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            hardwareInfo.forEach { (label, value) ->
                                InfoCard(label = label, value = value)
                            }
                        }
                    }
                }
            }

            item {
                Column(modifier = Modifier.fillMaxWidth()) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clickable { systemExpanded = !systemExpanded }
                            .padding(vertical = 8.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text("Current State")
                        Icon(
                            imageVector = if (systemExpanded) Icons.Default.ExpandLess else Icons.Default.ExpandMore,
                            contentDescription = null
                        )
                    }
                    AnimatedVisibility(visible = systemExpanded) {
                        val filteredState = systemState
                        Column(
                            verticalArrangement = Arrangement.spacedBy(8.dp),
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            filteredState.forEach { (label, value) ->
                                val history = viewModel.historyData.value[label] ?: emptyList()
                                val bounds = viewModel.getBoundsFor(label)
                                val color = when {
                                    label.contains("RAM", ignoreCase = true) -> Color(0xFF42A5F5)
                                    label.contains("Temp", ignoreCase = true) -> Color(0xFFFFA726)
                                    label.contains("Battery", ignoreCase = true) -> Color(0xFF9CCC65)
                                    else -> Color(0xFFBA68C8)
                                }
                                InfoCard(
                                    label = label,
                                    value = value,
                                    history = history,
                                    color = color,
                                    minValue = bounds.first,
                                    maxValue = bounds.second
                                )
                            }
                        }
                    }
                }
            }

            item { Spacer(modifier = Modifier.height(100.dp)) }
        }
    }
}

@Preview(showBackground = true)
@Composable
fun HomeScreenPreview() {
    SkynetMonitorTheme {
        HomeScreen(PaddingValues())
    }
}
