package fr.arthur.keusch.heimdall.services

import android.content.Context
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.Collections

object FallDetectionService {

    private const val WINDOW_MS = 10_000L
    private const val SAMPLING_MS = 20L
    private const val MODEL_FILE = "xgboost_pond_trial_has_fall.onnx"

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
                    synchronized(window) {
                        window.add(s)
                    }
                }
            }
        }

        loopJob = scope.launch {
            while (true) {
                delay(WINDOW_MS)

                val rawSamples = synchronized(window) {
                    val copy = window.toList()
                    window.clear()
                    copy
                }

                if (rawSamples.isNotEmpty()) {
                    val downsampled = mutableListOf<SensorSnapshot>()
                    var lastTimestamp = 0L

                    for (sample in rawSamples) {
                        if (sample.wallTimeMillis - lastTimestamp >= SAMPLING_MS) {
                            downsampled.add(sample)
                            lastTimestamp = sample.wallTimeMillis
                        }
                    }

                    logAverages(downsampled)

                    val finalDecision = fallDetection(downsampled)
                    _prediction.value = finalDecision
                }
            }
        }
    }

    private fun logAverages(samples: List<SensorSnapshot>) {
        if (samples.isEmpty()) return

        val count = samples.size.toFloat()
        var sumAx = 0f;
        var sumAy = 0f;
        var sumAz = 0f
        var sumGx = 0f;
        var sumGy = 0f;
        var sumGz = 0f
        var sumQw = 0f;
        var sumQx = 0f;
        var sumQy = 0f;
        var sumQz = 0f
        var sumVax = 0f;
        var sumVay = 0f;
        var sumVaz = 0f

        for (s in samples) {
            sumAx += s.ax ?: 0f; sumAy += s.ay ?: 0f; sumAz += s.az ?: 0f
            sumGx += s.gx ?: 0f; sumGy += s.gy ?: 0f; sumGz += s.gz ?: 0f
            sumQw += s.qw ?: 1f; sumQx += s.qx ?: 0f; sumQy += s.qy ?: 0f; sumQz += s.qz ?: 0f
            sumVax += s.vax ?: 0f; sumVay += s.vay ?: 0f; sumVaz += s.vaz ?: 0f
        }

        Log.i(
            "Heimdall-Fall", """
            --- MOYENNES FENÊTRE (N=$count) ---
            Accel (m/s²): X=${sumAx / count}, Y=${sumAy / count}, Z=${sumAz / count}
            Gyro (rad/s): X=${sumGx / count}, Y=${sumGy / count}, Z=${sumGz / count}
            Quat (w,x,y,z): W=${sumQw / count}, X=${sumQx / count}, Y=${sumQy / count}, Z=${sumQz / count}
            V-Accel: X=${sumVax / count}, Y=${sumVay / count}, Z=${sumVaz / count}
            -----------------------------------
        """.trimIndent()
        )
    }

    private suspend fun fallDetection(samples: List<SensorSnapshot>): Boolean {
        return withContext(Dispatchers.Default) {
            val session = ortSession ?: return@withContext false
            val votes = mutableListOf<Int>()

            try {
                for (s in samples) {
                    val features = floatArrayOf(
                        22f, 1.82f, 97f,
                        s.ax ?: 0f, s.ay ?: 0f, s.az ?: 0f,
                        s.gx ?: 0f, s.gy ?: 0f, s.gz ?: 0f,
                        s.qw ?: 1f, s.qx ?: 0f, s.qy ?: 0f, s.qz ?: 0f,
                        s.vax ?: 0f, s.vay ?: 0f, s.vaz ?: 0f
                    )

                    val inputTensor = OnnxTensor.createTensor(ortEnv, arrayOf(features))
                    val inputs = Collections.singletonMap("float_input", inputTensor)
                    val results = session.run(inputs)

                    val label = (results[0].value as LongArray)[0].toInt()
                    votes.add(label)

                    inputTensor.close()
                    results.close()
                }

                val countFall = votes.count { it == 1 }
                val countOk = votes.count { it == 0 }

                val finalDecision = countFall > countOk

                Log.i(
                    "Heimdall-Fall",
                    "RÉSULTAT : $countFall votes CHUTE / $countOk OK. Décision: $finalDecision"
                )

                return@withContext finalDecision
            } catch (e: Exception) {
                Log.e("Heimdall-Fall", "Erreur Inférence: ${e.message}")
            }
            false
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