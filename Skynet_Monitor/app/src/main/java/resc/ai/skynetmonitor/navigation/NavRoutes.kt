package resc.ai.skynetmonitor.navigation

sealed class NavRoutes(val route: String) {
    object Home : NavRoutes("home")
    object Stats : NavRoutes("stats")
}
