package resc.ai.skynetmonitor.ui.screens

import androidx.compose.animation.AnimatedContent
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.List
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import resc.ai.skynetmonitor.service.DatasetItem
import resc.ai.skynetmonitor.service.PerformanceSample
import resc.ai.skynetmonitor.service.PromptItem
import resc.ai.skynetmonitor.service.PromptResult
import resc.ai.skynetmonitor.ui.components.MiniGraph
import resc.ai.skynetmonitor.ui.theme.SkynetMonitorTheme
import resc.ai.skynetmonitor.viewmodel.StatsNavigation
import resc.ai.skynetmonitor.viewmodel.StatsViewModel
import java.util.Locale

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun StatsScreen(innerPadding: PaddingValues, viewModel: StatsViewModel = viewModel()) {
    val state by viewModel.state.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(innerPadding)
    ) {
        if (state.navigation == StatsNavigation.DatasetList || state.navigation == StatsNavigation.ModelComparison) {
            TabRow(selectedTabIndex = if (state.navigation == StatsNavigation.DatasetList) 0 else 1) {
                Tab(
                    selected = state.navigation == StatsNavigation.DatasetList,
                    onClick = { viewModel.navigateTo(StatsNavigation.DatasetList) },
                    text = { Text("Datasets") }
                )
                Tab(
                    selected = state.navigation == StatsNavigation.ModelComparison,
                    onClick = { viewModel.navigateTo(StatsNavigation.ModelComparison) },
                    text = { Text("Model Comparison") }
                )
            }
        } else {
            TopAppBar(
                title = { 
                    Text(
                        when (state.navigation) {
                            is StatsNavigation.DatasetStats -> "Dataset Stats"
                            is StatsNavigation.PromptList -> "Prompts"
                            is StatsNavigation.PromptDetail -> "Details"
                            else -> "Statistics"
                        }
                    ) 
                },
                navigationIcon = {
                    IconButton(onClick = { 
                        when (state.navigation) {
                            is StatsNavigation.DatasetStats, is StatsNavigation.PromptList -> viewModel.navigateTo(StatsNavigation.DatasetList)
                            is StatsNavigation.PromptDetail -> viewModel.navigateTo(StatsNavigation.PromptList((state.navigation as StatsNavigation.PromptDetail).dataset))
                            else -> {}
                        }
                    }) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                    }
                }
            )
        }

        if (state.navigation != StatsNavigation.DatasetList) {
            DeviceFilterChips(
                devices = state.devices,
                selectedId = state.selectedDeviceFilter,
                onSelected = { viewModel.setDeviceFilter(it) }
            )
        }

        Box(modifier = Modifier.weight(1f)) {
            AnimatedContent(targetState = state.navigation, label = "stats_nav") { nav ->
                when (nav) {
                    is StatsNavigation.DatasetList -> DatasetListScreen(state.datasets, viewModel)
                    is StatsNavigation.DatasetStats -> DatasetStatsScreen(nav.dataset, viewModel, state.selectedDeviceFilter)
                    is StatsNavigation.PromptList -> PromptListScreen(nav.dataset, viewModel)
                    is StatsNavigation.PromptDetail -> PromptDetailScreen(nav.prompt, viewModel, state.selectedDeviceFilter)
                    is StatsNavigation.ModelComparison -> ModelComparisonScreen(viewModel)
                }
            }
        }
    }
}

@Composable
fun DeviceFilterChips(devices: List<resc.ai.skynetmonitor.service.DeviceItem>, selectedId: Int?, onSelected: (Int?) -> Unit) {
    LazyColumn(modifier = Modifier.fillMaxWidth().heightIn(max = 60.dp)) {
        item {
            Row(
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                FilterChip(
                    selected = selectedId == null,
                    onClick = { onSelected(null) },
                    label = { Text("All Devices") }
                )
                devices.forEach { device ->
                    FilterChip(
                        selected = selectedId == device.id,
                        onClick = { onSelected(device.id) },
                        label = { Text(device.name) }
                    )
                }
            }
        }
    }
}

@Composable
fun DatasetListScreen(datasets: List<DatasetItem>, viewModel: StatsViewModel) {
    LazyColumn(contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        items(datasets) { dataset ->
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(dataset.name, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                    dataset.description?.let { Text(it, style = MaterialTheme.typography.bodyMedium) }
                    Spacer(Modifier.height(16.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(
                            onClick = { viewModel.navigateTo(StatsNavigation.DatasetStats(dataset)) },
                            modifier = Modifier.weight(1f)
                        ) {
                            Icon(Icons.Default.QueryStats, contentDescription = null)
                            Spacer(Modifier.width(4.dp))
                            Text("Stats")
                        }
                        OutlinedButton(
                            onClick = { viewModel.navigateTo(StatsNavigation.PromptList(dataset)) },
                            modifier = Modifier.weight(1f)
                        ) {
                            Icon(Icons.AutoMirrored.Filled.List, contentDescription = null)
                            Spacer(Modifier.width(4.dp))
                            Text("Prompts")
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun DatasetStatsScreen(dataset: DatasetItem, viewModel: StatsViewModel, deviceFilter: Int?) {
    val state by viewModel.state.collectAsState()
    var allResults by remember { mutableStateOf<List<PromptResult>>(emptyList()) }
    var avgSamples by remember { mutableStateOf<List<PerformanceSample>>(emptyList()) }
    var isLoading by remember { mutableStateOf(true) }

    LaunchedEffect(dataset) {
        val results = mutableListOf<PromptResult>()
        dataset.prompts.forEach { prompt ->
            results.addAll(viewModel.getResultsForPrompt(prompt.id))
        }
        allResults = results
        val samplesList = mutableListOf<List<PerformanceSample>>()
        results.take(5).forEach { result -> samplesList.add(viewModel.getPerformanceSamples(result.id)) }
        avgSamples = samplesList.maxByOrNull { it.size } ?: emptyList()
        isLoading = false
    }

    if (isLoading) {
        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { CircularProgressIndicator() }
    } else {
        val filteredResults = if (deviceFilter != null) allResults.filter { it.idDevices == deviceFilter } else allResults
        LazyColumn(contentPadding = PaddingValues(16.dp)) {
            item {
                Text("Overall Dataset Performance", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                Spacer(Modifier.height(16.dp))
            }
            item {
                StatCard("Avg. Tokens/sec", String.format(Locale.getDefault(), "%.2f", filteredResults.mapNotNull { it.responseTokensPerS }.average().takeIf { !it.isNaN() } ?: 0.0), Icons.Default.Speed, Color(0xFF4CAF50))
            }
            if (avgSamples.isNotEmpty()) {
                item {
                    Spacer(Modifier.height(24.dp))
                    Text("Resource Usage Trend (Last Session)", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                    Card(modifier = Modifier.fillMaxWidth().padding(vertical = 12.dp)) {
                        Column(Modifier.padding(16.dp)) {
                            Text("RAM Consumption (MB)", style = MaterialTheme.typography.labelSmall, color = Color(0xFF42A5F5))
                            MiniGraph(data = avgSamples.map { it.ramCurrentMb }, color = Color(0xFF42A5F5), minValue = 0f, maxValue = avgSamples.maxOf { it.ramMaxMb }.coerceAtLeast(1f))
                            Spacer(Modifier.height(16.dp))
                            Text("Battery Temp (°C)", style = MaterialTheme.typography.labelSmall, color = Color(0xFFFFA726))
                            MiniGraph(data = avgSamples.map { it.batteryTemperatureC }, color = Color(0xFFFFA726), minValue = 20f, maxValue = 50f)
                        }
                    }
                }
            }
            item {
                Spacer(Modifier.height(24.dp))
                Text("Performance by Model", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                Spacer(Modifier.height(8.dp))
            }
            val modelStats = filteredResults.groupBy { it.idModel }
            items(modelStats.keys.toList()) { modelId ->
                val modelName = state.models.find { it.id == modelId }?.name ?: "Model #$modelId"
                val resultsForModel = modelStats[modelId] ?: emptyList()
                val avgSpeed = resultsForModel.mapNotNull { it.responseTokensPerS }.average().takeIf { !it.isNaN() } ?: 0.0
                Card(Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
                    Column(Modifier.padding(16.dp)) {
                        Text(modelName, fontWeight = FontWeight.Bold)
                        LinearProgressIndicator(progress = { (avgSpeed / 50.0).toFloat().coerceIn(0f, 1f) }, modifier = Modifier.fillMaxWidth().height(8.dp).clip(CircleShape))
                        Text(String.format(Locale.getDefault(), "Avg Speed: %.2f tok/s", avgSpeed), style = MaterialTheme.typography.labelSmall)
                    }
                }
            }
        }
    }
}

@Composable
fun PromptListScreen(dataset: DatasetItem, viewModel: StatsViewModel) {
    LazyColumn(contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        items(dataset.prompts) { prompt ->
            Card(modifier = Modifier.fillMaxWidth().clickable { viewModel.navigateTo(StatsNavigation.PromptDetail(prompt, dataset)) }) {
                Row(modifier = Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
                    Text(prompt.prompt, maxLines = 1, modifier = Modifier.weight(1f))
                    Icon(Icons.Default.ChevronRight, contentDescription = null)
                }
            }
        }
    }
}

@Composable
fun PromptDetailScreen(prompt: PromptItem, viewModel: StatsViewModel, deviceFilter: Int?) {
    val state by viewModel.state.collectAsState()
    var results by remember { mutableStateOf<List<PromptResult>>(emptyList()) }
    var selectedResultId by remember { mutableStateOf<Int?>(null) }
    var samples by remember { mutableStateOf<List<PerformanceSample>>(emptyList()) }
    var isLoading by remember { mutableStateOf(true) }

    LaunchedEffect(prompt) {
        results = viewModel.getResultsForPrompt(prompt.id)
        isLoading = false
    }
    LaunchedEffect(selectedResultId) {
        samples = if (selectedResultId != null) viewModel.getPerformanceSamples(selectedResultId!!) else emptyList()
    }
    if (isLoading) {
        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { CircularProgressIndicator() }
    } else {
        val filteredResults = if (deviceFilter != null) results.filter { it.idDevices == deviceFilter } else results
        LazyColumn(contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            item {
                Text("Prompt Context", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.primary)
                Text(prompt.prompt, style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.Medium)
                Spacer(Modifier.height(8.dp)); HorizontalDivider()
            }
            items(filteredResults) { result ->
                val modelName = state.models.find { it.id == result.idModel }?.name ?: "Model #${result.idModel}"
                val deviceName = state.devices.find { it.id == result.idDevices }?.name ?: "Device #${result.idDevices}"
                Card(modifier = Modifier.fillMaxWidth().clickable { selectedResultId = if (selectedResultId == result.id) null else result.id }, colors = CardDefaults.cardColors(containerColor = if (selectedResultId == result.id) MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.5f) else MaterialTheme.colorScheme.surface), border = CardDefaults.outlinedCardBorder().takeIf { selectedResultId != result.id }) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                            Text(modelName, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.primary)
                            Text(deviceName, style = MaterialTheme.typography.labelSmall)
                        }
                        Spacer(Modifier.height(8.dp)); Text(result.response, style = MaterialTheme.typography.bodyMedium)
                        Row(Modifier.padding(top = 8.dp), horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                            result.responseTokensPerS?.let { LabelValue("Speed", String.format(Locale.getDefault(), "%.2f tok/s", it)) }
                            result.responseTimeMs?.let { LabelValue("Latency", "${it}ms") }
                        }
                        if (selectedResultId == result.id) {
                            if (samples.isNotEmpty()) {
                                Spacer(Modifier.height(16.dp)); Text("Hardware Impact", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
                                Spacer(Modifier.height(8.dp)); Text("RAM (MB)", style = MaterialTheme.typography.labelSmall, color = Color(0xFF42A5F5))
                                MiniGraph(data = samples.map { it.ramCurrentMb }, color = Color(0xFF42A5F5), minValue = 0f, maxValue = samples.maxOf { it.ramMaxMb }.coerceAtLeast(1f))
                                Spacer(Modifier.height(12.dp)); Text("Battery Temperature (°C)", style = MaterialTheme.typography.labelSmall, color = Color(0xFFFFA726))
                                MiniGraph(data = samples.map { it.batteryTemperatureC }, color = Color(0xFFFFA726), minValue = 20f, maxValue = 50f)
                            } else { Text("Loading performance samples...", style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(top = 16.dp)) }
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun ModelComparisonScreen(viewModel: StatsViewModel) {
    val state by viewModel.state.collectAsState()
    var results by remember { mutableStateOf<List<PromptResult>>(emptyList()) }
    var isLoading by remember { mutableStateOf(true) }

    LaunchedEffect(Unit) {
        results = viewModel.fetchAllResults()
        isLoading = false
    }

    if (isLoading) {
        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { CircularProgressIndicator() }
    } else {
        val filteredResults = if (state.selectedDeviceFilter != null) results.filter { it.idDevices == state.selectedDeviceFilter } else results
        val modelStats = filteredResults.groupBy { it.idModel }
        LazyColumn(contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            item {
                Text("Model Comparison", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
                Text("Comparing throughput across available hardware", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.outline)
                Spacer(Modifier.height(16.dp))
            }
            items(state.models) { model ->
                val resultsForModel = modelStats[model.id] ?: emptyList()
                val avgSpeed = resultsForModel.mapNotNull { it.responseTokensPerS }.average().takeIf { !it.isNaN() } ?: 0.0
                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                            Text(model.name, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                            Text(String.format(Locale.getDefault(), "%.2f tok/s", avgSpeed), color = MaterialTheme.colorScheme.primary, fontWeight = FontWeight.Bold)
                        }
                        Spacer(Modifier.height(8.dp)); LinearProgressIndicator(progress = { (avgSpeed / 50.0).toFloat().coerceIn(0f, 1f) }, modifier = Modifier.fillMaxWidth().height(8.dp).clip(CircleShape))
                        Spacer(Modifier.height(8.dp)); Text("${resultsForModel.size} benchmarks recorded", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.outline)
                    }
                }
            }
        }
    }
}

@Composable
fun StatCard(label: String, value: String, icon: ImageVector, color: Color) {
    Card(modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
        Row(modifier = Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
            Icon(icon, contentDescription = null, tint = color)
            Spacer(Modifier.width(16.dp))
            Column {
                Text(label, style = MaterialTheme.typography.labelMedium)
                Text(value, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
            }
        }
    }
}

@Composable
fun LabelValue(label: String, value: String) {
    Column {
        Text(label, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.outline)
        Text(value, style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.Bold)
    }
}
