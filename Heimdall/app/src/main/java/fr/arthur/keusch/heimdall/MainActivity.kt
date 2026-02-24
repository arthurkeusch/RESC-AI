package fr.arthur.keusch.heimdall

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.wear.compose.foundation.lazy.*
import androidx.wear.compose.material3.*
import androidx.wear.compose.material3.lazy.*
import fr.arthur.keusch.heimdall.services.*
import fr.arthur.keusch.heimdall.theme.HeimdallTheme
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter

class MainActivity : ComponentActivity() {

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val activityOk = permissions[Manifest.permission.ACTIVITY_RECOGNITION] ?: false
        val notificationOk = if (Build.VERSION.SDK_INT >= 33) {
            permissions[Manifest.permission.POST_NOTIFICATIONS] ?: false
        } else true

        if (activityOk && notificationOk) {
            FallDetectionForegroundService.start(applicationContext)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setShowWhenLocked(true)
        setTurnScreenOn(true)
        val km = getSystemService(KEYGUARD_SERVICE) as android.app.KeyguardManager
        km.requestDismissKeyguard(this, null)

        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        checkAndRequestPermissions()

        SensorsBus.start(applicationContext)
        FallDetectionForegroundService.start(applicationContext)

        setContent {
            WearApp()
        }
    }

    private fun checkAndRequestPermissions() {
        val permissionsNeeded = mutableListOf<String>()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACTIVITY_RECOGNITION)
            != PackageManager.PERMISSION_GRANTED
        ) {
            permissionsNeeded.add(Manifest.permission.ACTIVITY_RECOGNITION)
        }

        if (Build.VERSION.SDK_INT >= 33 && ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.POST_NOTIFICATIONS
            )
            != PackageManager.PERMISSION_GRANTED
        ) {
            permissionsNeeded.add(Manifest.permission.POST_NOTIFICATIONS)
        }

        if (permissionsNeeded.isNotEmpty()) {
            permissionLauncher.launch(permissionsNeeded.toTypedArray())
        }
    }
}

@Composable
fun WearApp() {
    val state by FallDetectionForegroundService.emergencyState.collectAsState()

    if (state != FallDetectionForegroundService.EmergencyState.IDLE) {
        BackHandler(enabled = true) { }
    }

    HeimdallTheme {
        AppScaffold {
            when (state) {
                FallDetectionForegroundService.EmergencyState.QUESTION -> EmergencyQuestionScreen()
                FallDetectionForegroundService.EmergencyState.ESCALATED -> EscalatedScreen()
                else -> MonitoringScreen()
            }
        }
    }
}

@Composable
fun EmergencyQuestionScreen() {
    val context = LocalContext.current
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(8.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(
            "CHUTE DÉTECTÉE",
            style = MaterialTheme.typography.titleMedium,
            color = Color.Red,
            textAlign = TextAlign.Center
        )
        Text(
            "Besoin d'aide ?",
            style = MaterialTheme.typography.bodySmall,
            modifier = Modifier.padding(bottom = 8.dp)
        )
        Button(modifier = Modifier.fillMaxWidth(), onClick = {
            val i = Intent(context, FallDetectionForegroundService::class.java).apply {
                action = FallDetectionForegroundService.ACTION_CONFIRM_HELP
            }
            context.startService(i)
        }) { Text("OUI") }
        Button(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 4.dp),
            colors = ButtonDefaults.buttonColors(containerColor = Color.DarkGray),
            onClick = {
                val i = Intent(context, FallDetectionForegroundService::class.java).apply {
                    action = FallDetectionForegroundService.ACTION_STOP_ALARM
                }
                context.startService(i)
            }) { Text("NON") }
    }
}

@Composable
fun EscalatedScreen() {
    val context = LocalContext.current
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(12.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(
            "ALERTE ENVOYÉE",
            style = MaterialTheme.typography.titleMedium,
            color = Color.Green,
            textAlign = TextAlign.Center
        )
        Text(
            "Les secours ont été prévenus.",
            style = MaterialTheme.typography.bodyMedium,
            textAlign = TextAlign.Center,
            modifier = Modifier.padding(vertical = 8.dp)
        )
        Button(modifier = Modifier.fillMaxWidth(), onClick = {
            val i = Intent(context, FallDetectionForegroundService::class.java).apply {
                action = FallDetectionForegroundService.ACTION_STOP_ALARM
            }
            context.startService(i)
        }) { Text("FERMER") }
    }
}

@Composable
fun MonitoringScreen() {
    val listState = rememberTransformingLazyColumnState()
    val transformationSpec = rememberTransformationSpec()
    ScreenScaffold(scrollState = listState) { contentPadding ->
        TransformingLazyColumn(contentPadding = contentPadding, state = listState) {
            item {
                ListHeader(
                    modifier = Modifier
                        .fillMaxWidth()
                        .transformedHeight(this, transformationSpec),
                    transformation = SurfaceTransformation(transformationSpec)
                ) {
                    Text("Heimdall")
                }
            }
            item { SensorReadings() }
        }
    }
}

@Composable
fun SensorReadings() {
    val lastFallEpoch by HealthEventBus.lastFall.collectAsState(initial = null)

    val fallFormatter = DateTimeFormatter.ofPattern("dd/MM/yyyy HH:mm:ss")
        .withZone(ZoneId.systemDefault())

    val fallText = lastFallEpoch?.let { fallFormatter.format(Instant.ofEpochMilli(it)) } ?: "—"

    Column(
        modifier = Modifier.fillMaxWidth(),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text("Dernière chute:", style = MaterialTheme.typography.labelMedium)
        Text(
            text = fallText,
            style = MaterialTheme.typography.bodyMedium,
            color = Color.Yellow,
            textAlign = TextAlign.Center
        )
    }
}