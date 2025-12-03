package resc.ai.skynetmonitor

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.pager.HorizontalPager
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.BarChart
import androidx.compose.material.icons.filled.Folder
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.launch
import resc.ai.skynetmonitor.navigation.NavRoutes
import resc.ai.skynetmonitor.ui.screens.HomeScreen
import resc.ai.skynetmonitor.ui.screens.StatsScreen
import resc.ai.skynetmonitor.ui.screens.SettingScreen
import resc.ai.skynetmonitor.ui.screens.DatabaseScreen
import resc.ai.skynetmonitor.ui.theme.SkynetMonitorTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            SkynetMonitorTheme {
                SkynetMonitorApp()
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class, ExperimentalFoundationApi::class)
@Composable
fun SkynetMonitorApp() {
    val pages = listOf(NavRoutes.Home, NavRoutes.Database, NavRoutes.Stats, NavRoutes.Setting)
    val pagerState = rememberPagerState(initialPage = 0, pageCount = { pages.size })
    val coroutineScope = rememberCoroutineScope()

    val colorScheme = MaterialTheme.colorScheme
    val selectedColor = colorScheme.primary.copy(alpha = 0.25f)
    val selectedContentColor = colorScheme.onPrimaryContainer
    val unselectedContentColor = colorScheme.onSurfaceVariant

    Scaffold(
        modifier = Modifier.fillMaxSize(), bottomBar = {
            NavigationBar(
                containerColor = colorScheme.surface, tonalElevation = 4.dp
            ) {
                pages.forEachIndexed { index, page ->
                    val selected = pagerState.currentPage == index

                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .padding(vertical = 6.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Box(
                            modifier = Modifier
                                .clip(MaterialTheme.shapes.large)
                                .background(if (selected) selectedColor else Color.Transparent)
                                .padding(horizontal = 12.dp, vertical = 8.dp),
                            contentAlignment = Alignment.Center
                        ) {
                            val iconColor =
                                if (selected) selectedContentColor else unselectedContentColor
                            val textColor =
                                if (selected) selectedContentColor else unselectedContentColor

                            Column(
                                horizontalAlignment = Alignment.CenterHorizontally
                            ) {
                                when (page) {
                                    NavRoutes.Home -> Icon(
                                        Icons.Filled.Home,
                                        contentDescription = "Home",
                                        tint = iconColor
                                    )

                                    NavRoutes.Stats -> Icon(
                                        Icons.Filled.BarChart,
                                        contentDescription = "Stats",
                                        tint = iconColor
                                    )

                                    NavRoutes.Database -> Icon(
                                        Icons.Filled.Folder,
                                        contentDescription = "Database",
                                        tint = iconColor
                                    )

                                    NavRoutes.Setting -> Icon(
                                        Icons.Filled.Settings,
                                        contentDescription = "Parameters",
                                        tint = iconColor
                                    )
                                }
                                Text(
                                    when (page) {
                                        NavRoutes.Home -> "Home"
                                        NavRoutes.Stats -> "Statistics"
                                        NavRoutes.Database -> "Database"
                                        NavRoutes.Setting -> "Settings"
                                    },
                                    style = MaterialTheme.typography.labelMedium,
                                    color = textColor
                                )
                            }
                        }
                        Box(
                            modifier = Modifier
                                .matchParentSize()
                                .background(Color.Transparent)
                                .clickable {
                                    coroutineScope.launch {
                                        pagerState.animateScrollToPage(index)
                                    }
                                })
                    }
                }
            }
        }) { innerPadding ->
        HorizontalPager(
            state = pagerState, modifier = Modifier.fillMaxSize()
        ) { page ->
            when (pages[page]) {
                NavRoutes.Home -> HomeScreen(innerPadding)
                NavRoutes.Stats -> StatsScreen(innerPadding)
                NavRoutes.Setting -> SettingScreen(innerPadding)
                NavRoutes.Database -> DatabaseScreen(innerPadding)
            }
        }
    }
}

@Preview(showBackground = true)
@Composable
fun SkynetMonitorAppPreview() {
    SkynetMonitorTheme {
        SkynetMonitorApp()
    }
}
