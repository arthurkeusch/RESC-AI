package fr.arthur.keusch.heimdall.services

import android.content.Context
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.nio.FloatBuffer
import java.util.Collections

object FallDetectionService {

    private const val WINDOW_MS = 10_000L
    private const val SAMPLING_MS = 20L
    private const val MODEL_FILE = "xgboost_pond_trial_has_fall.onnx"
    private const val NUM_FEATURES = 16

    private val _prediction = MutableStateFlow(false)
    val prediction = _prediction.asStateFlow()

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var collectJob: Job? = null
    private var loopJob: Job? = null

    private val window = mutableListOf<SensorSnapshot>()
    private val ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var ortSession: OrtSession? = null

    fun start(context: Context) {
        if (collectJob != null || loopJob != null) return

        scope.launch(Dispatchers.IO) {
            try {
                val modelBytes = context.assets.open(MODEL_FILE).readBytes()
                ortSession = ortEnv.createSession(modelBytes)
                Log.d("Heimdall-Fall", "Modèle ONNX chargé")
            } catch (e: Exception) {
                Log.e("Heimdall-Fall", "Erreur chargement: ${e.message}")
            }
        }

        SensorsBus.start(context.applicationContext)

        collectJob = scope.launch {
            SensorsBus.state.collect { s ->
                if (s.wallTimeMillis > 0L) {
                    synchronized(window) { window.add(s) }
                }
            }
        }

        loopJob = scope.launch {
            var nextWindowTime = System.currentTimeMillis() + WINDOW_MS
            while (true) {
                val now = System.currentTimeMillis()

                if (now >= nextWindowTime) {
                    val rawSamples = synchronized(window) {
                        val copy = window.toList()
                        window.clear()
                        copy
                    }

                    if (rawSamples.isNotEmpty()) {
                        val downsampledForAI = performDownsample(rawSamples)
                        val finalDecision = fallDetection(context, downsampledForAI)
                        _prediction.value = finalDecision
                    }
                    nextWindowTime = System.currentTimeMillis() + WINDOW_MS
                }
                delay(100)
            }
        }
    }

    private fun performDownsample(raw: List<SensorSnapshot>): List<SensorSnapshot> {
        val downsampled = mutableListOf<SensorSnapshot>()
        if (raw.isEmpty()) return downsampled

        var lastTimestamp = 0L
        for (sample in raw) {
            if (sample.wallTimeMillis - lastTimestamp >= SAMPLING_MS) {
                downsampled.add(sample)
                lastTimestamp = sample.wallTimeMillis
            }
        }
        return downsampled
    }

    private suspend fun fallDetection(context: Context, samples: List<SensorSnapshot>): Boolean {
        if (samples.isEmpty()) return false
        val session = ortSession ?: return false

        val prefs = context.getSharedPreferences("user_prefs", Context.MODE_PRIVATE)
        val userHeight = prefs.getFloat("height", 1.82f)
        val userWeight = prefs.getFloat("weight", 97.0f)

        return withContext(Dispatchers.Default) {
            try {
                val numRows = samples.size
                val flatArray = FloatArray(numRows * NUM_FEATURES)

                samples.forEachIndexed { i, s ->
                    val offset = i * NUM_FEATURES
                    flatArray[offset + 0] = userHeight
                    flatArray[offset + 1] = userWeight
                    flatArray[offset + 2] = i * (SAMPLING_MS / 1000f)
                    flatArray[offset + 3] = s.ax ?: 0f
                    flatArray[offset + 4] = s.ay ?: 0f
                    flatArray[offset + 5] = s.az ?: 0f
                    flatArray[offset + 6] = s.gx ?: 0f
                    flatArray[offset + 7] = s.gy ?: 0f
                    flatArray[offset + 8] = s.gz ?: 0f
                    flatArray[offset + 9] = s.qw ?: 1f
                    flatArray[offset + 10] = s.qx ?: 0f
                    flatArray[offset + 11] = s.qy ?: 0f
                    flatArray[offset + 12] = s.qz ?: 0f
                    flatArray[offset + 13] = s.vax ?: 0f
                    flatArray[offset + 14] = s.vay ?: 0f
                    flatArray[offset + 15] = s.vaz ?: 0f
                }

                val inputShape = longArrayOf(numRows.toLong(), NUM_FEATURES.toLong())
                val inputTensor =
                    OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(flatArray), inputShape)

                val inputs = Collections.singletonMap("float_input", inputTensor)
                session.run(inputs).use { results ->
                    val output = results[0].value
                    val predictions = when (output) {
                        is LongArray -> output
                        is Array<*> -> (output as Array<LongArray>).map { it[0] }.toLongArray()
                        else -> LongArray(0)
                    }

                    val countFall = predictions.count { it == 1L }
                    val ratio = countFall.toFloat() / numRows

                    Log.d(
                        "Heimdall-Fall",
                        "Inférence : $countFall detections / $numRows points (${
                            String.format(
                                "%.1f",
                                ratio * 100
                            )
                        }%)"
                    )
                    return@withContext ratio > 0.15f
                }
            } catch (e: Exception) {
                Log.e("Heimdall-Fall", "Inference Error: ${e.message}")
                false
            }
        }
    }

    fun stop() {
        collectJob?.cancel()
        loopJob?.cancel()
        collectJob = null
        loopJob = null
        try {
            ortSession?.close()
        } catch (e: Exception) {
        }
        synchronized(window) { window.clear() }
    }
}