package resc.ai.skynetmonitor.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.launch
import resc.ai.skynetmonitor.config.AppConfig
import resc.ai.skynetmonitor.ui.theme.SkynetMonitorTheme

@Composable
fun SettingScreen(innerPadding: PaddingValues) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val apiUrlFlow = AppConfig.apiUrl.collectAsState(initial = "")
    var apiUrl by remember { mutableStateOf("") }

    LaunchedEffect(apiUrlFlow.value) {
        apiUrl = apiUrlFlow.value
    }

    LaunchedEffect(Unit) {
        AppConfig.init(context)
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .padding(innerPadding),
        contentAlignment = Alignment.Center
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            TextField(
                value = apiUrl,
                onValueChange = { newValue ->
                    apiUrl = newValue
                    scope.launch { AppConfig.setApiUrl(context, newValue) }
                },
                modifier = Modifier.fillMaxWidth(),
                label = { Text("API URL") }
            )
        }
    }
}

@Preview(showBackground = true)
@Composable
fun ParamScreenPreview() {
    SkynetMonitorTheme {
        SettingScreen(PaddingValues())
    }
}
