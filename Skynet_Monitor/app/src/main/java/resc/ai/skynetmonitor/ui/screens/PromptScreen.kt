package resc.ai.skynetmonitor.ui.screens

import android.net.Uri
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
import kotlinx.coroutines.launch
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
    var name by remember { mutableStateOf("") }
    var description by remember { mutableStateOf("") }
    var isConversational by remember { mutableStateOf(false) }
    var jsonUri by remember { mutableStateOf<Uri?>(null) }
    var jsonLabel by remember { mutableStateOf("No file selected") }
    var isSubmitting by remember { mutableStateOf(false) }

    var datasets by remember { mutableStateOf<List<DatasetItem>>(emptyList()) }
    var refreshTrigger by remember { mutableStateOf(0) }

    val pickJson =
        rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
            if (uri != null) {
                jsonUri = uri
                jsonLabel = uri.lastPathSegment ?: "selected.json"
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
            title = { Text("Create prompt dataset") },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
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
                    Row(
                        verticalAlignment = Alignment.CenterVertically
                    ) {
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
                        Text(jsonLabel, modifier = Modifier.weight(1f))
                        OutlinedButton(onClick = { pickJson.launch(arrayOf("application/json")) }) {
                            Text("Select JSON")
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
    "prompt": "Your first prompt here"
  },
  {
    "prompt": "Another example prompt"
  }
]
                                """.trimIndent(),
                                style = MaterialTheme.typography.bodySmall
                            )
                        }
                    }
                }
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        if (!isSubmitting && jsonUri != null && name.isNotBlank()) {
                            isSubmitting = true
                            scope.launch {
                                val success = PromptService.importDataset(
                                    context = context,
                                    name = name,
                                    description = description,
                                    isConversational = isConversational,
                                    jsonUri = jsonUri!!
                                )
                                isSubmitting = false
                                if (success) {
                                    showDialog = false
                                    refreshTrigger++
                                }
                            }
                        }
                    }
                ) { Text(if (isSubmitting) "Processing..." else "Create") }
            },
            dismissButton = {
                if (!isSubmitting) {
                    TextButton(onClick = { showDialog = false }) { Text("Cancel") }
                }
            }
        )
    }
}
