package resc.ai.skynetmonitor.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import resc.ai.skynetmonitor.service.ModelService
import resc.ai.skynetmonitor.ui.theme.SkynetMonitorTheme
import resc.ai.skynetmonitor.viewmodel.DeviceInfoViewModel

@Composable
fun ParamScreen(innerPadding: PaddingValues, viewModel: DeviceInfoViewModel = viewModel()) {
    var apiUrl by remember { mutableStateOf(ModelService.API_BASE) }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .padding(innerPadding),
        contentAlignment = Alignment.Center
    ) {
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
                    viewModel.setApiUrl(newValue)
                },
                modifier = Modifier.weight(1f),
                label = {
                    Text("API:")
                },
            )
        }
    }
}

@Preview(showBackground = true)
@Composable
fun ParamScreenPreview() {
    SkynetMonitorTheme {
        ParamScreen(PaddingValues())
    }
}
