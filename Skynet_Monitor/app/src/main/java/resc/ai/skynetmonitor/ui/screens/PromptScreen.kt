package resc.ai.skynetmonitor.ui.screens

import android.net.Uri
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import resc.ai.skynetmonitor.service.DatasetItem
import resc.ai.skynetmonitor.service.PromptService
import resc.ai.skynetmonitor.viewmodel.DeviceInfoViewModel
import androidx.lifecycle.viewmodel.compose.viewModel
import resc.ai.skynetmonitor.ui.components.DatasetCard

@Composable
fun PromptScreen(innerPadding: PaddingValues, viewModel: DeviceInfoViewModel = viewModel()) {
    val scope = rememberCoroutineScope()
    val context = viewModel.ctx

    var showDialog by remember { mutableStateOf(false) }
    var selectedTabIndex by remember { mutableStateOf(0) } // 0 = Single, 1 = Batch

    // Single mode states
    var name by remember { mutableStateOf("") }
    var description by remember { mutableStateOf("") }
    var isConversational by remember { mutableStateOf(false) }
    var jsonUri by remember { mutableStateOf<Uri?>(null) }
    var jsonLabel by remember { mutableStateOf("No file selected") }

    // Batch mode states
    var batchUris by remember { mutableStateOf<List<Uri>>(emptyList()) }
    
    // Progress states
    var currentPromptProgress by remember { mutableStateOf(0) }
    var totalPromptsInDataset by remember { mutableStateOf(0) }
    var currentDatasetProgress by remember { mutableStateOf(0) }
    var totalDatasetsToProcess by remember { mutableStateOf(0) }
    var currentProcessingFileName by remember { mutableStateOf("") }
    
    var isSubmitting by remember { mutableStateOf(false) }
    var importJob by remember { mutableStateOf<kotlinx.coroutines.Job?>(null) }
    val createdDatasetIds = remember { mutableStateListOf<Int>() }

    var datasets by remember { mutableStateOf<List<DatasetItem>>(emptyList()) }
    var refreshTrigger by remember { mutableStateOf(0) }

    val pickJson =
        rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
            if (uri != null) {
                jsonUri = uri
                jsonLabel = uri.lastPathSegment ?: "selected.json"

                scope.launch(Dispatchers.IO) {
                    try {
                        context.contentResolver.openInputStream(uri)?.use { stream ->
                            val content = stream.bufferedReader().use { it.readText() }
                            val jsonArray = JSONArray(content)
                            
                            var foundName: String? = null
                            var foundDesc: String? = null
                            var foundConv: Boolean? = null

                            for (i in 0 until jsonArray.length()) {
                                val obj = jsonArray.getJSONObject(i)
                                if (obj.has("dataset")) foundName = obj.getString("dataset")
                                if (obj.has("description")) foundDesc = obj.getString("description")
                                if (obj.has("isConversational")) foundConv = obj.getBoolean("isConversational")
                                if (foundName != null || foundDesc != null || foundConv != null) break
                            }

                            withContext(Dispatchers.Main) {
                                foundName?.let { name = it }
                                foundDesc?.let { description = it }
                                foundConv?.let { isConversational = it }
                            }
                        }
                    } catch (e: Exception) {
                        Log.e("PromptScreen", "Metadata extraction failed", e)
                    }
                }
            }
        }

    val pickMultipleJson =
        rememberLauncherForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
            if (uris.isNotEmpty()) {
                batchUris = uris
            }
        }

    LaunchedEffect(refreshTrigger) {
        val result = PromptService.fetchDatasets(context)
        if (result != null) datasets = result
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .padding(innerPadding)
    ) {
        if (datasets.isEmpty()) {
            Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text("No datasets available")
            }
        } else {
            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                items(datasets.size) { index ->
                    DatasetCard(
                        dataset = datasets[index],
                        onDelete = { datasetToDelete ->
                            scope.launch {
                                val success =
                                    PromptService.deleteDataset(context, datasetToDelete.id)
                                if (success) {
                                    refreshTrigger++
                                }
                            }
                        },
                        onUpdated = {
                            refreshTrigger++
                        }
                    )
                }
            }
        }

        FloatingActionButton(
            onClick = {
                name = ""
                description = ""
                isConversational = false
                jsonUri = null
                jsonLabel = "No file selected"
                batchUris = emptyList()
                selectedTabIndex = 0
                isSubmitting = false
                showDialog = true
            },
            modifier = Modifier
                .align(Alignment.BottomEnd)
                .padding(20.dp)
        ) {
            Icon(Icons.Filled.Add, contentDescription = null)
        }
    }

    if (showDialog) {
        AlertDialog(
            onDismissRequest = { if (!isSubmitting) showDialog = false },
            title = { 
                Text(if (isSubmitting) "Importing Data..." else "Add prompt datasets") 
            },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    if (isSubmitting) {
                        // Progress UI
                        if (selectedTabIndex == 1) { // Batch mode
                            Text(
                                "Datasets: $currentDatasetProgress / $totalDatasetsToProcess",
                                style = MaterialTheme.typography.titleSmall
                            )
                            LinearProgressIndicator(
                                progress = { if (totalDatasetsToProcess > 0) currentDatasetProgress.toFloat() / totalDatasetsToProcess else 0f },
                                modifier = Modifier.fillMaxWidth()
                            )
                            Spacer(modifier = Modifier.height(8.dp))
                        }

                        Text(
                            "Current: $currentProcessingFileName",
                            style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.primary
                        )
                        Text(
                            "Prompts: $currentPromptProgress / $totalPromptsInDataset",
                            style = MaterialTheme.typography.bodySmall
                        )
                        LinearProgressIndicator(
                            progress = { if (totalPromptsInDataset > 0) currentPromptProgress.toFloat() / totalPromptsInDataset else 0f },
                            modifier = Modifier.fillMaxWidth()
                        )

                    } else {
                        // Config UI
                        TabRow(selectedTabIndex = selectedTabIndex) {
                            Tab(
                                selected = selectedTabIndex == 0,
                                onClick = { selectedTabIndex = 0 },
                                text = { Text("Single") }
                            )
                            Tab(
                                selected = selectedTabIndex == 1,
                                onClick = { selectedTabIndex = 1 },
                                text = { Text("Batch") }
                            )
                        }

                        Spacer(modifier = Modifier.height(8.dp))

                        if (selectedTabIndex == 0) {
                            OutlinedTextField(
                                value = name,
                                onValueChange = { name = it },
                                singleLine = true,
                                label = { Text("Dataset name") },
                                modifier = Modifier.fillMaxWidth()
                            )
                            OutlinedTextField(
                                value = description,
                                onValueChange = { description = it },
                                label = { Text("Description") },
                                modifier = Modifier.fillMaxWidth()
                            )
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Checkbox(
                                    checked = isConversational,
                                    onCheckedChange = { isConversational = it })
                                Text("Is conversational")
                            }
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.SpaceBetween,
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Text(jsonLabel, modifier = Modifier.weight(1f), maxLines = 1)
                                OutlinedButton(onClick = { pickJson.launch(arrayOf("application/json")) }) {
                                    Text("Select JSON")
                                }
                            }
                        } else {
                            Text(
                                "Batch import: 'dataset' and 'description' are mandatory in each JSON file.",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.secondary
                            )
                            Button(
                                onClick = { pickMultipleJson.launch(arrayOf("application/json")) },
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Text(if (batchUris.isEmpty()) "Select JSON Files" else "${batchUris.size} files selected")
                            }
                            if (batchUris.isNotEmpty()) {
                                Card(
                                    modifier = Modifier.fillMaxWidth(),
                                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
                                ) {
                                    Column(modifier = Modifier.padding(8.dp)) {
                                        batchUris.take(5).forEach { uri ->
                                            Text("• ${uri.lastPathSegment}", style = MaterialTheme.typography.labelSmall)
                                        }
                                        if (batchUris.size > 5) {
                                            Text("... and ${batchUris.size - 5} more", style = MaterialTheme.typography.labelSmall)
                                        }
                                    }
                                }
                            }
                        }

                        Card(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(top = 8.dp),
                            colors = CardDefaults.cardColors(
                                containerColor = MaterialTheme.colorScheme.surfaceVariant
                            )
                        ) {
                            Column(modifier = Modifier.padding(12.dp)) {
                                Text(
                                    "Expected JSON format:",
                                    style = MaterialTheme.typography.titleSmall
                                )
                                Spacer(modifier = Modifier.height(4.dp))
                                Text(
                                    text = """
[
  {
    "dataset": "Dataset Name", // Required in Batch
    "description": "Description here", // Required in Batch
    "isConversational": true // Optional (default: false)
  },
  { "prompt": "First prompt..." },
  { "prompt": "Second prompt..." }
]
                                    """.trimIndent(),
                                    style = MaterialTheme.typography.bodySmall.copy(
                                        fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace,
                                        fontSize = 10.sp
                                    ),
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                                Spacer(modifier = Modifier.height(8.dp))
                                Text(
                                    "Note: In 'Single' mode, metadata will auto-fill the form if present.",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.primary
                                )
                            }
                        }
                    }
                }
            },
            confirmButton = {
                if (!isSubmitting) {
                    TextButton(
                        onClick = {
                            if (selectedTabIndex == 0) {
                                if (jsonUri != null && name.isNotBlank()) {
                                    isSubmitting = true
                                    currentProcessingFileName = jsonLabel
                                    createdDatasetIds.clear()
                                    importJob = scope.launch {
                                        val success = PromptService.importDataset(
                                            context = context,
                                            name = name,
                                            description = description,
                                            isConversational = isConversational,
                                            jsonUri = jsonUri!!,
                                            onProgress = { current, total ->
                                                currentPromptProgress = current
                                                totalPromptsInDataset = total
                                            },
                                            onDatasetCreated = { createdDatasetIds.add(it) }
                                        )
                                        isSubmitting = false
                                        importJob = null
                                        if (success) {
                                            showDialog = false
                                            refreshTrigger++
                                        }
                                    }
                                }
                            } else {
                                if (batchUris.isNotEmpty()) {
                                    isSubmitting = true
                                    totalDatasetsToProcess = batchUris.size
                                    currentDatasetProgress = 0
                                    createdDatasetIds.clear()
                                    
                                    importJob = scope.launch {
                                        for (uri in batchUris) {
                                            currentProcessingFileName = uri.lastPathSegment ?: "Processing..."
                                            
                                            var finalName: String? = null
                                            var finalDesc: String? = null
                                            var finalConv = false

                                            try {
                                                context.contentResolver.openInputStream(uri)?.use { stream ->
                                                    val content = stream.bufferedReader().use { it.readText() }
                                                    val jsonArray = JSONArray(content)
                                                    for (i in 0 until jsonArray.length()) {
                                                        val obj = jsonArray.getJSONObject(i)
                                                        if (obj.has("dataset")) finalName = obj.getString("dataset")
                                                        if (obj.has("description")) finalDesc = obj.getString("description")
                                                        if (obj.has("isConversational")) finalConv = obj.getBoolean("isConversational")
                                                    }
                                                }
                                            } catch (e: Exception) { Log.e("PromptScreen", "Batch extraction error", e) }

                                            val nameToUse = finalName
                                            val descToUse = finalDesc
                                            if (nameToUse != null && descToUse != null) {
                                                PromptService.importDataset(
                                                    context = context,
                                                    name = nameToUse,
                                                    description = descToUse,
                                                    isConversational = finalConv,
                                                    jsonUri = uri,
                                                    onProgress = { current, total ->
                                                        currentPromptProgress = current
                                                        totalPromptsInDataset = total
                                                    },
                                                    onDatasetCreated = { createdDatasetIds.add(it) }
                                                )
                                            } else {
                                                Log.w("PromptScreen", "Skipping file $currentProcessingFileName due to missing metadata")
                                            }
                                            currentDatasetProgress++
                                        }
                                        isSubmitting = false
                                        importJob = null
                                        showDialog = false
                                        refreshTrigger++
                                    }
                                }
                            }
                        }
                    ) { Text("Create") }
                } else {
                    TextButton(
                        onClick = {
                            importJob?.cancel()
                            isSubmitting = false
                            importJob = null
                            // Cleanup: delete what was uploaded
                            scope.launch {
                                createdDatasetIds.forEach { id ->
                                    PromptService.deleteDataset(context, id)
                                }
                                createdDatasetIds.clear()
                                refreshTrigger++
                            }
                        },
                        colors = ButtonDefaults.textButtonColors(contentColor = MaterialTheme.colorScheme.error)
                    ) {
                        Text("Cancel Import")
                    }
                }
            },
            dismissButton = {
                if (!isSubmitting) {
                    TextButton(onClick = { showDialog = false }) { Text("Cancel") }
                }
            }
        )
    }
}
