package fr.arthur.keusch.heimdall.services

import android.annotation.SuppressLint
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.media.AudioAttributes
import android.media.MediaPlayer
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.util.Log
import androidx.core.app.NotificationCompat
import fr.arthur.keusch.heimdall.MainActivity
import fr.arthur.keusch.heimdall.R
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import androidx.core.net.toUri

class FallDetectionForegroundService : Service() {

    enum class EmergencyState { IDLE, QUESTION, ESCALATED }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var escalationJob: Job? = null
    private var awaitingResponse = false
    private var mediaPlayer: MediaPlayer? = null
    private var vibrator: Vibrator? = null

    override fun onCreate() {
        super.onCreate()
        vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val vibratorManager = getSystemService(VIBRATOR_MANAGER_SERVICE) as VibratorManager
            vibratorManager.defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            getSystemService(VIBRATOR_SERVICE) as Vibrator
        }

        ensureChannels()
        startForeground(ONGOING_ID, buildOngoingNotification())

        SensorsBus.start(applicationContext)
        FallDetectionService.start(applicationContext)

        scope.launch {
            FallDetectionService.prediction.collect { isFall ->
                if (isFall) triggerEmergencyFlow()
            }
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP_ALARM -> {
                _emergencyState.value = EmergencyState.IDLE
                cancelEmergency()
            }

            ACTION_CONFIRM_HELP -> {
                _emergencyState.value = EmergencyState.ESCALATED
                stopAlarmAndVibration()
                escalationJob?.cancel()
            }
        }
        return START_STICKY
    }

    private fun triggerEmergencyFlow() {
        if (awaitingResponse) return
        awaitingResponse = true
        _emergencyState.value = EmergencyState.QUESTION

        HealthEventBus.setLastFall(System.currentTimeMillis())

        playAlarmSound()
        startIntenseVibration()
        wakeScreen()
        showFullScreenTrigger()

        startEscalationTimer()
    }

    @SuppressLint("FullScreenIntentPolicy", "WearRecents")
    private fun showFullScreenTrigger() {
        val intent = Intent(this, MainActivity::class.java).apply {
            addFlags(
                Intent.FLAG_ACTIVITY_NEW_TASK or
                        Intent.FLAG_ACTIVITY_REORDER_TO_FRONT or
                        Intent.FLAG_ACTIVITY_SINGLE_TOP
            )
        }

        val pendingIntent = PendingIntent.getActivity(
            this, 0, intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val builder = NotificationCompat.Builder(this, CHANNEL_ALERT)
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setContentTitle("Urgence")
            .setContentText("Réponse requise")
            .setPriority(NotificationCompat.PRIORITY_MAX)
            .setCategory(NotificationCompat.CATEGORY_ALARM)
            .setFullScreenIntent(pendingIntent, true)
            .setOngoing(true)
            .setAutoCancel(false)

        val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        nm.notify(TRIGGER_ID, builder.build())
    }

    private fun playAlarmSound() {
        try {
            val soundUri = "android.resource://$packageName/${R.raw.heimdall_alarm_long}".toUri()
            mediaPlayer?.release()
            mediaPlayer = MediaPlayer().apply {
                setDataSource(applicationContext, soundUri)
                setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_ALARM)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                        .build()
                )
                isLooping = true
                prepare()
                start()
            }
        } catch (e: Exception) {
            Log.e("Heimdall", "Erreur son: ${e.message}")
        }
    }

    private fun startIntenseVibration() {
        vibrator?.let {
            if (it.hasVibrator()) {
                val pattern = longArrayOf(0, 500, 200, 500)
                it.vibrate(VibrationEffect.createWaveform(pattern, 0))
            }
        }
    }

    private fun stopAlarmAndVibration() {
        mediaPlayer?.let { if (it.isPlaying) it.stop(); it.release() }
        mediaPlayer = null
        vibrator?.cancel()
    }

    private fun cancelEmergency() {
        awaitingResponse = false
        escalationJob?.cancel()
        escalationJob = null
        stopAlarmAndVibration()
        val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        nm.cancel(TRIGGER_ID)
    }

    private fun buildOngoingNotification(): Notification {
        return NotificationCompat.Builder(this, CHANNEL_ONGOING)
            .setSmallIcon(R.drawable.splash_icon)
            .setContentTitle("Heimdall")
            .setContentText("Surveillance active")
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
    }

    private fun wakeScreen() {
        val pm = getSystemService(POWER_SERVICE) as PowerManager
        val wl = pm.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK,
            "Heimdall:fall_wake"
        )
        if (!wl.isHeld) wl.acquire(5_000L)
    }

    private fun ensureChannels() {
        val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        nm.createNotificationChannel(
            NotificationChannel(
                CHANNEL_ONGOING,
                "Heimdall Running",
                NotificationManager.IMPORTANCE_LOW
            )
        )

        val alertChannel = NotificationChannel(
            CHANNEL_ALERT,
            "Heimdall Trigger",
            NotificationManager.IMPORTANCE_HIGH
        ).apply {
            setSound(null, null)
            enableVibration(true)
        }
        nm.createNotificationChannel(alertChannel)
    }

    override fun onDestroy() {
        stopAlarmAndVibration(); scope.cancel(); super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun startEscalationTimer() {
        escalationJob?.cancel()
        escalationJob = scope.launch {
            val startTime = System.currentTimeMillis()
            val timeout = 30_000L

            while (awaitingResponse) {
                if (_emergencyState.value == EmergencyState.QUESTION) {
                    showFullScreenTrigger()
                    wakeScreen()
                }

                if (System.currentTimeMillis() - startTime >= timeout) {
                    _emergencyState.value = EmergencyState.ESCALATED
                    stopAlarmAndVibration()
                    awaitingResponse = false
                    val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
                    nm.cancel(TRIGGER_ID)
                    break
                }
                delay(2000L)
            }
        }
    }

    companion object {
        private const val CHANNEL_ONGOING = "heimdall_ongoing"
        private const val CHANNEL_ALERT = "heimdall_trigger"
        private const val ONGOING_ID = 9001
        private const val TRIGGER_ID = 9004
        const val ACTION_STOP_ALARM = "heimdall.action.STOP_ALARM"
        const val ACTION_CONFIRM_HELP = "heimdall.action.CONFIRM_HELP"
        private val _emergencyState = MutableStateFlow(EmergencyState.IDLE)
        val emergencyState = _emergencyState.asStateFlow()

        fun start(context: Context) {
            val i = Intent(context, FallDetectionForegroundService::class.java)
            context.startForegroundService(i)
        }
    }
}