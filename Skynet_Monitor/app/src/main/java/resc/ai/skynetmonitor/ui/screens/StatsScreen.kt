package resc.ai.skynetmonitor.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import resc.ai.skynetmonitor.ui.theme.SkynetMonitorTheme

@Composable
fun StatsScreen(innerPadding: PaddingValues) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .padding(innerPadding),
        contentAlignment = Alignment.Center
    ) {
        Text("Hello World - Stats Page")
    }
}

@Preview(showBackground = true)
@Composable
fun StatsScreenPreview() {
    SkynetMonitorTheme {
        StatsScreen(PaddingValues())
    }
}
