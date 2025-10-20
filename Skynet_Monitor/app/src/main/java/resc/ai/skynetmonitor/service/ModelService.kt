package resc.ai.skynetmonitor.service

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import org.json.JSONArray
import java.io.File
import java.net.URL
import javax.net.ssl.HttpsURLConnection

data class RemoteModel(
    val name: String,
    val params: String,
    val sizeBytes: Long,
    val filename: String
)

data class DownloadState(
    val name: String,
    val bytesReceived: Long,
    val totalBytes: Long,
    val speedBytesPerSec: Long,
    val etaSeconds: Long,
    val progress: Int
)

object ModelService {
    const val API_BASE = "https://resc-ai.arthur-keusch.fr:3000"
    private const val TAG = "ModelService"

    private var apiBase = API_BASE

    suspend fun fetchRemoteModels(): List<RemoteModel> = withContext(Dispatchers.IO) {
        try {
            val conn = (URL(apiBase + "/models").openConnection() as HttpsURLConnection).apply {
                requestMethod = "GET"
                connectTimeout = 7000
                readTimeout = 15000
            }
            val code = conn.responseCode

            if (code != 200) {
                val errorMsg = conn.errorStream?.bufferedReader()?.use { it.readText() }
                    ?: "No server error message"
                Log.e(TAG, "fetchRemoteModels() HTTP $code: $errorMsg")
                return@withContext emptyList()
            }

            val response = conn.inputStream.bufferedReader().use { it.readText() }
            val arr = JSONArray(response)
            List(arr.length()) { i ->
                val o = arr.getJSONObject(i)
                RemoteModel(
                    name = o.getString("name"),
                    params = o.getString("params"),
                    sizeBytes = o.optLong("size", -1L),
                    filename = o.getString("filename")
                )
            }
        } catch (e: Exception) {
            Log.e(TAG, "Exception in fetchRemoteModels()", e)
            return@withContext emptyList()
        }
    }

    fun resolveLocalFile(context: Context, filename: String): File {
        val dir = context.getDir("models", Context.MODE_PRIVATE)
        if (!dir.exists()) dir.mkdirs()
        return File(dir, filename)
    }

    fun setApiUrl(newUrl: String) {
        apiBase = newUrl
        Log.d(TAG, "API base URL set to $apiBase")
    }

    fun isModelDownloaded(context: Context, filename: String): Boolean {
        val f = resolveLocalFile(context, filename)
        return f.exists() && f.length() > 0
    }

    fun getLocalModelPath(ctx: Context, filename: String): String {
        return resolveLocalFile(ctx, filename).absolutePath
    }

    suspend fun downloadModel(
        context: Context,
        model: RemoteModel,
        onProgress: (DownloadState) -> Unit
    ): File = withContext(Dispatchers.IO) {
        try {
            val url = URL("$API_BASE/download/${model.name}")
            val conn = (url.openConnection() as HttpsURLConnection).apply {
                requestMethod = "GET"
                connectTimeout = 10000
                readTimeout = 60000
            }
            val code = conn.responseCode
            if (code != 200) {
                val errorMsg = conn.errorStream?.bufferedReader()?.use { it.readText() }
                    ?: "No server error message"
                throw IllegalStateException("Download failed with HTTP $code: $errorMsg")
            }

            val total = if (conn.contentLengthLong > 0) conn.contentLengthLong else model.sizeBytes
            val target = resolveLocalFile(context, model.filename)
            val tmp = File(target.parentFile, "${target.name}.part")

            tmp.outputStream().use { out ->
                conn.inputStream.use { ins ->
                    val buf = ByteArray(DEFAULT_BUFFER_SIZE)
                    var received = 0L
                    val last = System.nanoTime()
                    var lastTick = System.nanoTime()
                    while (true) {
                        val read = ins.read(buf)
                        if (read == -1) break
                        out.write(buf, 0, read)
                        received += read
                        val now = System.nanoTime()
                        val tickMs = (now - lastTick) / 1_000_000
                        if (tickMs >= 500L) {
                            val elapsedSec = ((now - last) / 1_000_000_000.0).coerceAtLeast(0.001)
                            val speed = (received / elapsedSec).toLong()
                            val remain =
                                if (total > 0 && received <= total) total - received else -1L
                            val eta = if (speed > 0 && remain > 0) (remain / speed) else -1L
                            val progress = if (total > 0) ((received * 100) / total).toInt() else 0
                            onProgress(
                                DownloadState(
                                    name = model.name,
                                    bytesReceived = received,
                                    totalBytes = if (total > 0) total else -1L,
                                    speedBytesPerSec = speed,
                                    etaSeconds = eta,
                                    progress = progress.coerceIn(0, 100)
                                )
                            )
                            lastTick = now
                        }
                    }
                }
            }
            if (target.exists()) target.delete()
            tmp.renameTo(target)
            delay(50)
            target
        } catch (e: Exception) {
            Log.e(TAG, "Exception in downloadModel()", e)
            throw e
        }
    }
}
