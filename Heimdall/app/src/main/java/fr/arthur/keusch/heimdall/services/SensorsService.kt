package fr.arthur.keusch.heimdall.services

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.SystemClock
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import kotlin.math.max
import kotlin.math.sqrt

const val SENSORS_BUFFER_MS: Long = 10_000L

data class SensorSnapshot(
    val wallTimeMillis: Long = 0L,
    val timeText: String = "—",

    val ax: Float? = null,
    val ay: Float? = null,
    val az: Float? = null,
    val aNorm: Float? = null,

    val gx: Float? = null,
    val gy: Float? = null,
    val gz: Float? = null,

    val qw: Float? = null,
    val qx: Float? = null,
    val qy: Float? = null,
    val qz: Float? = null,

    val vax: Float? = null,
    val vay: Float? = null,
    val vaz: Float? = null
)

object SensorsBus {

    private val _state = MutableStateFlow(SensorSnapshot())
    val state = _state.asStateFlow()

    private val _buffer = MutableStateFlow<List<SensorSnapshot>>(emptyList())
    val buffer = _buffer.asStateFlow()

    private var started = false
    private var sensorManager: SensorManager? = null
    private var listener: SensorEventListener? = null

    private val rotMat = FloatArray(9)
    private val quat = FloatArray(4)

    private var ax: Float? = null
    private var ay: Float? = null
    private var az: Float? = null

    private var gx: Float? = null
    private var gy: Float? = null
    private var gz: Float? = null

    private var qw: Float? = null
    private var qx: Float? = null
    private var qy: Float? = null
    private var qz: Float? = null

    private var vax: Float? = null
    private var vay: Float? = null
    private var vaz: Float? = null

    private var bootTimeMillis: Long = 0L
    private val timeFormatter =
        DateTimeFormatter.ofPattern("HH:mm:ss.SSS").withZone(ZoneId.systemDefault())

    private val ring = ArrayDeque<SensorSnapshot>()

    fun start(context: Context, delay: Int = SensorManager.SENSOR_DELAY_GAME) {
        if (started) return
        started = true

        sensorManager =
            context.applicationContext.getSystemService(Context.SENSOR_SERVICE) as SensorManager
        bootTimeMillis = System.currentTimeMillis() - SystemClock.elapsedRealtime()

        val accel = sensorManager!!.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        val gyro = sensorManager!!.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        val rot = sensorManager!!.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
        val linAccel = sensorManager!!.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)

        listener = object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent) {
                val wallMillis = bootTimeMillis + (event.timestamp / 1_000_000L)
                val timeText = timeFormatter.format(Instant.ofEpochMilli(wallMillis))

                when (event.sensor.type) {
                    Sensor.TYPE_ACCELEROMETER -> {
                        ax = event.values.getOrNull(0)
                        ay = event.values.getOrNull(1)
                        az = event.values.getOrNull(2)
                        updateVerticalFromAccelAndRotMat()
                    }

                    Sensor.TYPE_GYROSCOPE -> {
                        gx = event.values.getOrNull(0)
                        gy = event.values.getOrNull(1)
                        gz = event.values.getOrNull(2)
                    }

                    Sensor.TYPE_ROTATION_VECTOR -> {
                        SensorManager.getRotationMatrixFromVector(rotMat, event.values)
                        SensorManager.getQuaternionFromVector(quat, event.values)
                        qw = quat[0]; qx = quat[1]; qy = quat[2]; qz = quat[3]
                        updateVerticalFromAccelAndRotMat()
                    }

                    Sensor.TYPE_LINEAR_ACCELERATION -> {
                        vax = event.values.getOrNull(0)
                        vay = event.values.getOrNull(1)
                        vaz = event.values.getOrNull(2)
                    }
                }

                val snap = SensorSnapshot(
                    wallTimeMillis = wallMillis,
                    timeText = timeText,

                    ax = ax, ay = ay, az = az,
                    aNorm = norm(ax, ay, az),

                    gx = gx, gy = gy, gz = gz,

                    qw = qw, qx = qx, qy = qy, qz = qz,

                    vax = vax, vay = vay, vaz = vaz
                )

                _state.value = snap
                pushToBuffer(snap)
            }

            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
        }

        if (accel != null) sensorManager!!.registerListener(listener, accel, delay)
        if (gyro != null) sensorManager!!.registerListener(listener, gyro, delay)
        if (rot != null) sensorManager!!.registerListener(listener, rot, delay)
        if (linAccel != null) sensorManager!!.registerListener(listener, linAccel, delay)
    }

    fun stop() {
        if (!started) return
        started = false
        sensorManager?.unregisterListener(listener)
        listener = null
        sensorManager = null
        ring.clear()
        _buffer.value = emptyList()
    }

    private fun pushToBuffer(snap: SensorSnapshot) {
        ring.addLast(snap)
        trimBuffer(snap.wallTimeMillis)
        _buffer.value = ring.toList()
    }

    private fun trimBuffer(nowMillis: Long) {
        val cutoff = max(0L, nowMillis - SENSORS_BUFFER_MS)
        while (true) {
            val first = ring.firstOrNull() ?: break
            if (first.wallTimeMillis >= cutoff) break
            ring.removeFirst()
        }
    }

    private fun updateVerticalFromAccelAndRotMat() {
        if (ax == null || ay == null || az == null) return
        if (rotMat.all { it == 0f }) return
        val aX = ax!!
        val aY = ay!!
        val aZ = az!!
        val wx = rotMat[0] * aX + rotMat[1] * aY + rotMat[2] * aZ
        val wy = rotMat[3] * aX + rotMat[4] * aY + rotMat[5] * aZ
        val wz = rotMat[6] * aX + rotMat[7] * aY + rotMat[8] * aZ
        vax = wx
        vay = wy
        vaz = wz
    }

    private fun norm(x: Float?, y: Float?, z: Float?): Float? {
        if (x == null || y == null || z == null) return null
        return sqrt(x * x + y * y + z * z)
    }
}
