package resc.ai.skynetmonitor.service

import android.annotation.SuppressLint
import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import kotlinx.coroutines.isActive
import kotlinx.coroutines.CancellationException
import org.json.JSONArray
import java.io.File
import java.net.URL
import javax.net.ssl.HttpsURLConnection

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
            val conn = (URL("$apiBase/models").openConnection() as HttpsURLConnection).apply {
                requestMethod = "GET"
                connectTimeout = 7000
                readTimeout = 15000
            }
            val code = conn.responseCode
            if (code != 200) return@withContext emptyList()

            val response = conn.inputStream.bufferedReader().use { it.readText() }
            val arr = JSONArray(response)

            List(arr.length()) { i ->
                val o = arr.getJSONObject(i)
                val id = o.getLong("id")
                val name = o.getString("name")
                val filename = o.getString("filename")
                val size = o.optLong("size", -1L)
                RemoteModel(
                    id = id,
                    name = name,
                    filename = filename,
                    sizeBytes = size,
                    isLocal = false
                )
            }
        } catch (e: Exception) {
            Log.e(TAG, "Exception in fetchRemoteModels()", e)
            emptyList()
        }
    }

    @SuppressLint("DefaultLocale")
    fun formatSize(bytes: Long): String {
        if (bytes <= 0) return "Unknown"
        val kb = bytes / 1024.0
        val mb = kb / 1024.0
        val gb = mb / 1024.0
        return when {
            gb >= 1 -> String.format("%.2f GB", gb)
            mb >= 1 -> String.format("%.1f MB", mb)
            kb >= 1 -> String.format("%.0f KB", kb)
            else -> "$bytes B"
        }
    }

    fun resolveLocalFile(context: Context, filename: String): File {
        val dir = context.getDir("models", Context.MODE_PRIVATE)
        if (!dir.exists()) dir.mkdirs()
        return File(dir, filename)
    }

    fun setApiUrl(newUrl: String) {
        apiBase = newUrl
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
        var conn: HttpsURLConnection? = null
        try {
            val url = URL("$apiBase/models/download/${model.id}")
            conn = (url.openConnection() as HttpsURLConnection).apply {
                requestMethod = "GET"
                connectTimeout = 10000
                readTimeout = 5000
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
                    var lastTick = System.nanoTime()
                    val start = System.nanoTime()

                    while (true) {
                        if (!coroutineContext.isActive) throw CancellationException("Cancelled by user")
                        val read = ins.read(buf)
                        if (read == -1) break
                        out.write(buf, 0, read)
                        received += read

                        val now = System.nanoTime()
                        val elapsedSec = ((now - start) / 1_000_000_000.0).coerceAtLeast(0.001)
                        val tickMs = (now - lastTick) / 1_000_000

                        if (tickMs >= 500L) {
                            val speed = (received / elapsedSec).toLong()
                            val remaining =
                                if (total > 0 && received <= total) total - received else -1L
                            val eta = if (speed > 0 && remaining > 0) (remaining / speed) else -1L
                            val progress = if (total > 0) ((received * 100) / total).toInt() else 0

                            onProgress(
                                DownloadState(
                                    name = model.name,
                                    bytesReceived = received,
                                    totalBytes = total,
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
        } catch (e: CancellationException) {
            try {
                conn?.disconnect()
            } catch (_: Exception) {
            }
            throw e
        } catch (e: Exception) {
            Log.e(TAG, "Exception in downloadModel()", e)
            throw e
        }
    }
}
