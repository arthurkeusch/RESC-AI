package resc.ai.skynetmonitor.ui.screens

import android.net.Uri
import android.provider.OpenableColumns
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import resc.ai.skynetmonitor.viewmodel.RagViewModel

@Composable
fun RagScreen(innerPadding: PaddingValues, viewModel: RagViewModel = viewModel()) {
    val indexedFiles by viewModel.indexedFiles.collectAsState()
    val ragInfo by viewModel.ragInfo.collectAsState()
    val searchResults by viewModel.searchResults.collectAsState()
    val isInitializing by viewModel.isInitializing.collectAsState()
    val isProcessing by viewModel.isProcessing.collectAsState()
    val progress by viewModel.progress.collectAsState()

    val context = LocalContext.current
    var searchQuery by remember { mutableStateOf("") }

    val filePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            val fileName = getFileName(context, it) ?: "unknown_file.pdf"
            viewModel.addFile(it, fileName)
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .padding(innerPadding)
    ) {
        if (isInitializing) {
            Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    if (progress > 0f) {
                        CircularProgressIndicator(progress = { progress })
                    } else {
                        CircularProgressIndicator()
                    }
                    Spacer(Modifier.height(8.dp))
                    Text("Initializing RAG Engine...")
                    if (progress > 0f) {
                        Text("${(progress * 100).toInt()}%", style = MaterialTheme.typography.labelSmall)
                    }
                }
            }
        } else {
            Row(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(8.dp)
            ) {
                // Left Part: Search and Results
                Column(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxHeight()
                        .padding(end = 8.dp)
                ) {
                    OutlinedTextField(
                        value = searchQuery,
                        onValueChange = {
                            searchQuery = it
                            viewModel.search(it)
                        },
                        label = { Text("Search in RAG") },
                        modifier = Modifier.fillMaxWidth(),
                        leadingIcon = { Icon(Icons.Default.Search, contentDescription = null) }
                    )

                    Spacer(modifier = Modifier.height(16.dp))

                    Text("References:", style = MaterialTheme.typography.titleMedium)

                    LazyColumn(modifier = Modifier.fillMaxSize()) {
                        items(searchResults) { result ->
                            Card(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(vertical = 4.dp)
                            ) {
                                Text(
                                    text = result,
                                    modifier = Modifier.padding(8.dp),
                                    style = MaterialTheme.typography.bodySmall
                                )
                            }
                        }
                    }
                }

                VerticalDivider()

                // Right Part: Files and Info
                Column(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxHeight()
                        .padding(start = 8.dp)
                ) {
                    Text("RAG Info", style = MaterialTheme.typography.titleMedium)
                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 8.dp),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
                    ) {
                        Text(
                            text = ragInfo,
                            modifier = Modifier.padding(16.dp),
                            style = MaterialTheme.typography.bodyMedium
                        )
                    }

                    if (isProcessing) {
                        LinearProgressIndicator(
                            progress = { progress },
                            modifier = Modifier.fillMaxWidth()
                        )
                        Text("Processing file...", style = MaterialTheme.typography.labelSmall)
                    }

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text("Indexed Files:", style = MaterialTheme.typography.titleMedium)
                        IconButton(onClick = { viewModel.clearIndex() }) {
                            Icon(Icons.Default.Refresh, contentDescription = "Reset RAG", tint = MaterialTheme.colorScheme.error)
                        }
                    }

                    LazyColumn(modifier = Modifier.fillMaxSize()) {
                        items(indexedFiles) { fileName ->
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(vertical = 4.dp),
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.SpaceBetween
                            ) {
                                Text(
                                    text = fileName,
                                    modifier = Modifier.weight(1f),
                                    style = MaterialTheme.typography.bodyMedium
                                )
                                IconButton(onClick = { viewModel.removeFile(fileName) }) {
                                    Icon(Icons.Default.Delete, contentDescription = "Delete", tint = MaterialTheme.colorScheme.outline)
                                }
                            }
                            HorizontalDivider()
                        }
                    }
                }
            }

            // Explicit Floating Action Button for adding files
            FloatingActionButton(
                onClick = { filePickerLauncher.launch("application/pdf") },
                modifier = Modifier
                    .align(Alignment.BottomEnd)
                    .padding(16.dp)
            ) {
                Icon(Icons.Default.Add, contentDescription = "Add PDF")
            }
        }
    }
}

private fun getFileName(context: android.content.Context, uri: Uri): String? {
    var result: String? = null
    if (uri.scheme == "content") {
        context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) {
                val index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (index != -1) {
                    result = cursor.getString(index)
                }
            }
        }
    }
    if (result == null) {
        result = uri.path
        val cut = result?.lastIndexOf('/') ?: -1
        if (cut != -1) {
            result = result?.substring(cut + 1)
        }
    }
    return result
}
