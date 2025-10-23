package resc.ai.skynetmonitor.ui.components

import androidx.compose.animation.animateContentSize
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.ExpandLess
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.ElevatedButton
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.launch
import resc.ai.skynetmonitor.service.DatasetItem
import resc.ai.skynetmonitor.service.PromptItem
import resc.ai.skynetmonitor.service.PromptService

@Composable
fun DatasetCard(
    dataset: DatasetItem,
    onDelete: (DatasetItem) -> Unit,
    modifier: Modifier = Modifier,
    onUpdated: () -> Unit = {}
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var expanded by remember { mutableStateOf(false) }
    var showConfirmDatasetDelete by remember { mutableStateOf(false) }
    var showEditDataset by remember { mutableStateOf(false) }
    var showAddPromptDialog by remember { mutableStateOf(false) }

    var localName by remember(dataset.id) { mutableStateOf(dataset.name) }
    var localDescription by remember(dataset.id) { mutableStateOf(dataset.description ?: "") }
    var localConversational by remember(dataset.id) { mutableStateOf(dataset.isConversational) }

    if (showConfirmDatasetDelete) {
        AlertDialog(
            onDismissRequest = { showConfirmDatasetDelete = false },
            title = { Text("Confirmation") },
            text = { Text("Are you sure you want to delete this dataset?") },
            confirmButton = {
                Button(onClick = {
                    showConfirmDatasetDelete = false
                    onDelete(dataset)
                }) { Text("Delete") }
            },
            dismissButton = {
                TextButton(onClick = { showConfirmDatasetDelete = false }) { Text("Cancel") }
            }
        )
    }

    if (showEditDataset) {
        EditDatasetDialog(
            name = localName,
            description = localDescription,
            isConversational = localConversational,
            onDismiss = { showEditDataset = false },
            onConfirm = { newName, newDesc, newConv ->
                scope.launch {
                    val ok = PromptService.updateDataset(
                        context = context,
                        datasetId = dataset.id,
                        name = newName,
                        description = newDesc,
                        isConversational = newConv
                    )
                    if (ok) {
                        localName = newName
                        localDescription = newDesc
                        localConversational = newConv
                        onUpdated()
                    }
                    showEditDataset = false
                }
            }
        )
    }

    if (showAddPromptDialog) {
        EditPromptDialog(
            initialText = "",
            title = "Add Prompt",
            confirmLabel = "Add",
            onDismiss = { showAddPromptDialog = false },
            onConfirm = { text ->
                scope.launch {
                    val ok = PromptService.addPrompt(
                        context = context,
                        datasetId = dataset.id,
                        promptText = text
                    )
                    if (ok) onUpdated()
                    showAddPromptDialog = false
                }
            }
        )
    }

    Card(
        modifier = modifier
            .fillMaxWidth()
            .animateContentSize()
            .clickable { expanded = !expanded },
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        localName,
                        style = MaterialTheme.typography.titleLarge,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                    if (localDescription.isNotBlank()) {
                        Spacer(Modifier.height(4.dp))
                        Text(
                            localDescription,
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            maxLines = 2,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                }
                Icon(
                    imageVector = if (expanded) Icons.Filled.ExpandLess else Icons.Filled.ExpandMore,
                    contentDescription = null
                )
            }

            if (expanded) {
                Spacer(Modifier.height(12.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    if (localConversational) {
                        Box(
                            modifier = Modifier
                                .clip(MaterialTheme.shapes.small)
                                .background(MaterialTheme.colorScheme.primary.copy(alpha = 0.12f))
                                .padding(horizontal = 6.dp, vertical = 2.dp)
                        ) {
                            Text(
                                text = "Conversational",
                                color = MaterialTheme.colorScheme.primary,
                                fontSize = 10.sp
                            )
                        }
                    }
                    Spacer(modifier = Modifier.weight(1f))
                    Row(
                        horizontalArrangement = Arrangement.spacedBy(4.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            imageVector = Icons.Filled.Edit,
                            contentDescription = "Edit dataset",
                            modifier = Modifier.size(24.dp)
                        )
                        Icon(
                            imageVector = Icons.Filled.Delete,
                            contentDescription = "Delete dataset",
                            tint = Color(0xFFFF0000),
                            modifier = Modifier.size(24.dp)
                        )
                    }
                }

                Spacer(Modifier.height(12.dp))

                dataset.prompts.forEachIndexed { idx, prompt ->
                    PromptRow(
                        prompt = prompt,
                        onEdited = { newText ->
                            scope.launch {
                                val ok = PromptService.updatePrompt(
                                    context = context,
                                    promptId = prompt.id,
                                    newText = newText,
                                    datasetId = dataset.id
                                )
                                if (ok) onUpdated()
                            }
                        },
                        onDeleted = {
                            scope.launch {
                                val ok = PromptService.deletePrompt(
                                    context = context,
                                    promptId = prompt.id
                                )
                                if (ok) onUpdated()
                            }
                        }
                    )
                    if (idx < dataset.prompts.size - 1) Spacer(Modifier.height(4.dp))
                }

                Spacer(Modifier.height(8.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.End
                ) {
                    ElevatedButton(onClick = { showAddPromptDialog = true }) {
                        Icon(Icons.Filled.Add, contentDescription = null)
                        Spacer(Modifier.width(6.dp))
                        Text("Add prompt")
                    }
                }
            }
        }
    }
}

@Composable
private fun PromptRow(
    prompt: PromptItem,
    onEdited: (String) -> Unit,
    onDeleted: () -> Unit
) {
    var showEdit by remember { mutableStateOf(false) }
    var showDeleteConfirm by remember { mutableStateOf(false) }

    if (showEdit) {
        EditPromptDialog(
            initialText = prompt.prompt,
            onDismiss = { showEdit = false },
            onConfirm = { newText ->
                onEdited(newText)
                showEdit = false
            }
        )
    }

    if (showDeleteConfirm) {
        AlertDialog(
            onDismissRequest = { showDeleteConfirm = false },
            title = { Text("Delete prompt") },
            text = { Text("Are you sure you want to delete this prompt?") },
            confirmButton = {
                Button(onClick = {
                    showDeleteConfirm = false
                    onDeleted()
                }) { Text("Delete") }
            },
            dismissButton = {
                TextButton(onClick = { showDeleteConfirm = false }) { Text("Cancel") }
            }
        )
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 12.dp, vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = prompt.prompt,
                style = MaterialTheme.typography.bodyMedium,
                modifier = Modifier.weight(2f),
                maxLines = 3,
                overflow = TextOverflow.Ellipsis
            )
            Row(
                horizontalArrangement = Arrangement.spacedBy(4.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    imageVector = Icons.Filled.Edit,
                    contentDescription = "Edit prompt",
                    modifier = Modifier
                        .size(24.dp)
                        .clickable { showEdit = true }
                )
                Icon(
                    imageVector = Icons.Filled.Delete,
                    contentDescription = "Delete prompt",
                    tint = Color(0xFFFF0000),
                    modifier = Modifier
                        .size(24.dp)
                        .clickable { showDeleteConfirm = true }
                )
            }
        }
    }
}

@Composable
private fun EditDatasetDialog(
    name: String,
    description: String,
    isConversational: Boolean,
    onDismiss: () -> Unit,
    onConfirm: (String, String, Boolean) -> Unit
) {
    var localName by remember { mutableStateOf(name) }
    var localDescription by remember { mutableStateOf(description) }
    var localConv by remember { mutableStateOf(isConversational) }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Edit dataset") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                OutlinedTextField(
                    value = localName,
                    onValueChange = { localName = it },
                    singleLine = true,
                    label = { Text("Dataset name") },
                    modifier = Modifier.fillMaxWidth()
                )
                OutlinedTextField(
                    value = localDescription,
                    onValueChange = { localDescription = it },
                    label = { Text("Description") },
                    modifier = Modifier.fillMaxWidth()
                )
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Checkbox(checked = localConv, onCheckedChange = { localConv = it })
                    Text("Is conversational")
                }
            }
        },
        confirmButton = {
            TextButton(
                onClick = { onConfirm(localName.trim(), localDescription.trim(), localConv) },
                enabled = localName.isNotBlank()
            ) { Text("Save") }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) { Text("Cancel") }
        }
    )
}

@Composable
private fun EditPromptDialog(
    initialText: String,
    onDismiss: () -> Unit,
    onConfirm: (String) -> Unit,
    title: String = "Edit Prompt",
    confirmLabel: String = "Save"
) {
    var text by remember { mutableStateOf(initialText) }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(title) },
        text = {
            Column {
                OutlinedTextField(
                    value = text,
                    onValueChange = { text = it },
                    label = { Text("Prompt") },
                    modifier = Modifier.fillMaxWidth(),
                    minLines = 3
                )
            }
        },
        confirmButton = {
            TextButton(
                onClick = { onConfirm(text.trim()) },
                enabled = text.isNotBlank()
            ) { Text(confirmLabel) }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) { Text("Cancel") }
        }
    )
}
