package resc.ai.skynetmonitor.service

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.widget.Toast
import android.util.Log
import java.io.File
import java.text.DecimalFormat
import java.util.concurrent.Executors

object ModelLauncherService {

    private const val TAG = "SkynetModel"

    // Un petit pool pour éviter de bloquer l’UI
    private val io = Executors.newSingleThreadExecutor()
    private val main = Handler(Looper.getMainLooper())

    // --- Helpers debug -------------------------------------------------------

    private fun logD(msg: String) {
        Log.d(TAG, msg)
    }

    private fun logE(msg: String, tr: Throwable? = null) {
        Log.e(TAG, msg, tr)
    }

    private fun toast(context: Context, msg: String) {
        main.post { Toast.makeText(context.applicationContext, msg, Toast.LENGTH_SHORT).show() }
    }

    private fun humanSize(bytes: Long): String {
        if (bytes < 0) return "—"
        val kb = bytes / 1024.0
        val mb = kb / 1024.0
        val gb = mb / 1024.0
        val df = DecimalFormat("#,##0.00")
        return when {
            gb >= 1 -> "${df.format(gb)} GB"
            mb >= 1 -> "${df.format(mb)} MB"
            else    -> "${df.format(kb)} KB"
        }
    }

    // --- API publique --------------------------------------------------------

    /**
     * Démarre le modèle en natif.
     * Ajoute des logs détaillés + toasts pour diagnostiquer facilement.
     */
    fun startModel(
        context: Context,
        modelPath: String,
        onOutput: (String) -> Unit,
        onReady: () -> Unit,
        onError: (String) -> Unit
    ) {
        logD("startModel() CALLED with path='$modelPath'")
        toast(context, "▶️ StartModel…")

        io.execute {
            try {
                // 1) Vérifier le fichier
                val file = File(modelPath)
                logD("Checking file existence at '$modelPath'")
                if (!file.exists()) {
                    val msg = "Model file NOT FOUND at: $modelPath"
                    logE(msg)
                    main.post {
                        onError(msg)
                        toast(context, "❌ $msg")
                    }
                    return@execute
                }

                val len = file.length()
                logD("Model file found. size=${len}B (${humanSize(len)}) name='${file.name}' abs='${file.absolutePath}'")
                if (len <= 0L) {
                    val msg = "Model file is EMPTY (0 bytes): ${file.absolutePath}"
                    logE(msg)
                    main.post {
                        onError(msg)
                        toast(context, "❌ $msg")
                    }
                    return@execute
                }

                // 2) Vérifier le chargement de la lib JNI (au cas où)
                try {
                    // Déclenche la classe pour forcer System.loadLibrary("llama") (dans NativeLlama.init{}).
                    Class.forName("resc.ai.skynetmonitor.service.NativeLlama")
                    logD("NativeLlama class loaded OK (System.loadLibrary likely succeeded)")
                } catch (cnfe: ClassNotFoundException) {
                    val msg = "NativeLlama class NOT FOUND (package/class mismatch?)"
                    logE(msg, cnfe)
                    main.post {
                        onError(msg)
                        toast(context, "❌ $msg")
                    }
                    return@execute
                } catch (t: Throwable) {
                    val msg = "Error while loading NativeLlama class: ${t.message}"
                    logE(msg, t)
                    main.post {
                        onError(msg)
                        toast(context, "❌ $msg")
                    }
                    return@execute
                }

                // 3) Appeler le natif
                logD("Calling NativeLlama.initModel(...)")
                main.post { onOutput("🔧 Loading model: ${file.name} …") }

                val ok: Boolean = try {
                    NativeLlama.initModel(file.absolutePath)
                } catch (ue: UnsatisfiedLinkError) {
                    val msg = "UnsatisfiedLinkError in initModel (JNI signature/lib mismatch): ${ue.message}"
                    logE(msg, ue)
                    main.post {
                        onError(msg)
                        onOutput("❌ $msg")
                        toast(context, "❌ JNI error (initModel)")
                    }
                    return@execute
                } catch (t: Throwable) {
                    val msg = "Throwable in initModel: ${t.message}"
                    logE(msg, t)
                    main.post {
                        onError(msg)
                        onOutput("❌ $msg")
                        toast(context, "❌ Native error (initModel)")
                    }
                    return@execute
                }

                logD("NativeLlama.initModel returned: $ok")
                if (!ok) {
                    val msg = "initModel returned FALSE (model not loaded)"
                    logE(msg)
                    main.post {
                        onError(msg)
                        onOutput("❌ $msg")
                        toast(context, "❌ $msg")
                    }
                    return@execute
                }

                // 4) Succès
                main.post {
                    logD("Model READY ✅")
                    onOutput("✅ Model loaded: ${file.name}")
                    toast(context, "✅ Model loaded")
                    onReady()
                }
            } catch (t: Throwable) {
                val msg = "startModel() fatal error: ${t.message}"
                logE(msg, t)
                main.post {
                    onError(msg)
                    onOutput("❌ $msg")
                    toast(context, "❌ $msg")
                }
            }
        }
    }

    /**
     * Envoie un prompt au modèle (appel natif generate).
     * Ajoute des logs détaillés + toasts en cas d’erreur.
     */
    fun sendPrompt(prompt: String, onResponse: (String) -> Unit) {
        logD("sendPrompt() CALLED with prompt='${prompt.take(200)}${if (prompt.length > 200) "…" else ""}'")

        io.execute {
            val res: String = try {
                logD("Calling NativeLlama.generate(...)")
                NativeLlama.generate(prompt) ?: "❌ Native returned null"
            } catch (ue: UnsatisfiedLinkError) {
                val msg = "UnsatisfiedLinkError in generate (JNI mismatch): ${ue.message}"
                logE(msg, ue)
                "❌ $msg"
            } catch (t: Throwable) {
                val msg = "Throwable in generate: ${t.message}"
                logE(msg, t)
                "❌ $msg"
            }

            logD("generate() returned: '${res.take(200)}${if (res.length > 200) "…" else ""}'")
            main.post { onResponse(res) }
        }
    }

    /**
     * Libère le modèle natif.
     */
    fun stopModel(context: Context? = null) {
        logD("stopModel() CALLED")
        io.execute {
            try {
                NativeLlama.freeModel()
                logD("freeModel() OK")
                context?.let { toast(it, "🧹 Model freed") }
            } catch (ue: UnsatisfiedLinkError) {
                logE("UnsatisfiedLinkError in freeModel: ${ue.message}", ue)
            } catch (t: Throwable) {
                logE("Throwable in freeModel: ${t.message}", t)
            }
        }
    }
}
