package resc.ai.skynetmonitor.navigation

sealed class NavRoutes(val route: String) {
    object Home : NavRoutes("home")
    object Rag : NavRoutes("rag")
    object Stats : NavRoutes("stats")
    object Setting : NavRoutes("setting")
    object Database : NavRoutes("database")
}
