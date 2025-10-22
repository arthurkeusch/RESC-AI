package resc.ai.skynetmonitor.ui.screens

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier

@Composable
fun DatabaseScreen(innerPadding: PaddingValues) {
    val tabs = listOf("Models", "Prompts")
    var selectedTab by remember { mutableIntStateOf(0) }

    Column(
        modifier = Modifier
            .statusBarsPadding()
    ) {
        TabRow(selectedTabIndex = selectedTab) {
            tabs.forEachIndexed { index, title ->
                Tab(
                    selected = selectedTab == index,
                    onClick = { selectedTab = index },
                    text = { Text(title) }
                )
            }
        }

        when (selectedTab) {
            0 -> ModelScreen(innerPadding)
            1 -> PromptScreen(innerPadding)
        }
    }
}
