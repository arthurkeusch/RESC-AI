package resc.ai.skynetmonitor.ui.components

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import resc.ai.skynetmonitor.viewmodel.DeviceInfoViewModel

@Composable
fun ModelChatDialog(
    viewModel: DeviceInfoViewModel,
    onClose: () -> Unit
) {
    val chat = viewModel.benchmarkState.collectAsState()
    var userInput by remember { mutableStateOf("") }

    Dialog(onDismissRequest = { viewModel.stopBenchmark(); onClose() }) {
        Surface(
            modifier = Modifier.fillMaxSize(),
            color = MaterialTheme.colorScheme.surface,
            shape = MaterialTheme.shapes.medium
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(16.dp),
                verticalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = "Chat with ${chat.value.modelName}",
                    style = MaterialTheme.typography.titleLarge
                )

                Spacer(Modifier.height(8.dp))

                LazyColumn(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth()
                        .padding(4.dp)
                ) {
                    items(chat.value.output) { line ->
                        Text(text = line, style = MaterialTheme.typTypography().bodyMedium)
                        Spacer(Modifier.height(6.dp))
                    }
                }

                Spacer(Modifier.height(8.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    OutlinedTextField(
                        value = userInput,
                        onValueChange = { userInput = it },
                        modifier = Modifier.weight(1f),
                        placeholder = { Text("Ask something...") }
                    )
                    Button(
                        onClick = {
                            if (userInput.isNotBlank()) {
                                viewModel.sendPrompt(userInput)
                                userInput = ""
                            }
                        }
                    ) {
                        Text("Send")
                    }
                }

                Spacer(Modifier.height(8.dp))

                Button(
                    onClick = { viewModel.stopBenchmark(); onClose() },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Close")
                }
            }
        }
    }
}

@Composable
private fun MaterialTheme.typTypography() = this.typography
