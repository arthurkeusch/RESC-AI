package fr.arthur.keusch.heimdall.services

import android.util.Log
import androidx.health.services.client.PassiveListenerService
import androidx.health.services.client.data.HealthEvent
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow

object HealthEventBus {
    private val _lastFall = MutableStateFlow<Long?>(null)
    val lastFall = _lastFall.asStateFlow()

    fun setLastFall(epochMillis: Long) {
        _lastFall.value = epochMillis
    }
}

class PassiveHealthEventService : PassiveListenerService() {
    override fun onHealthEventReceived(event: HealthEvent) {
        if (event.type == HealthEvent.Type.FALL_DETECTED) {
            val t = event.eventTime.toEpochMilli()
            HealthEventBus.setLastFall(t)
            Log.i("Heimdall", "FALL_DETECTED (System) at $t")
        }
        super.onHealthEventReceived(event)
    }
}