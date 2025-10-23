package resc.ai.skynetmonitor.service

import android.annotation.SuppressLint
import android.content.Context
import android.database.Cursor
import android.net.Uri
import android.provider.OpenableColumns
import android.util.Log
import android.widget.Toast
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import org.json.JSONArray
import resc.ai.skynetmonitor.config.AppConfig
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
    private const val TAG = "ModelService"

    private suspend fun showToast(context: Context, message: String) {
        withContext(Dispatchers.Main) {
            Toast.makeText(context, "Error : $message", Toast.LENGTH_LONG).show()
        }
    }

    private suspend fun getApiBase(context: Context): String {
        return AppConfig.apiUrl.first()
    }

    suspend fun fetchRemoteModels(context: Context): List<RemoteModel> =
        withContext(Dispatchers.IO) {
            try {
                val apiBase = getApiBase(context)
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
                    RemoteModel(
                        id = o.getLong("id_model"),
                        name = o.getString("name"),
                        filename = o.getString("filename"),
                        sizeBytes = o.optLong("size", -1L),
                        params = o.optString("params", ""),
                        isLocal = false
                    )
                }
            } catch (e: Exception) {
                Log.e(TAG, "Exception in fetchRemoteModels()", e)
                try {
                    showToast(context, e.message ?: "Unknown error")
                } catch (_: Exception) {
                }
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

    fun isModelDownloaded(context: Context, filename: String): Boolean {
        val f = resolveLocalFile(context, filename)
        return f.exists() && f.length() > 0
    }

    fun getLocalModelPath(ctx: Context, filename: String): String {
        return resolveLocalFile(ctx, filename).absolutePath
    }

    suspend fun deleteRemoteModel(context: Context, model: RemoteModel) =
        withContext(Dispatchers.IO) {
            try {
                val apiBase = getApiBase(context)
                val url = URL("$apiBase/models/${model.id}")
                val conn = (url.openConnection() as HttpsURLConnection).apply {
                    requestMethod = "DELETE"
                    connectTimeout = 10000
                    readTimeout = 15000
                }
                val code = conn.responseCode
                if (code !in 200..299) {
                    val msg =
                        conn.errorStream?.bufferedReader()?.use { it.readText() } ?: "Delete failed"
                    try {
                        showToast(context, "HTTP $code : $msg")
                    } catch (_: Exception) {
                    }
                    throw IllegalStateException("Delete failed with HTTP $code: $msg")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Exception in deleteRemoteModel()", e)
                try {
                    showToast(context, e.message ?: "Unknown error")
                } catch (_: Exception) {
                }
                throw e
            }
        }

    suspend fun downloadModel(
        context: Context,
        model: RemoteModel,
        onProgress: (DownloadState) -> Unit
    ): File = withContext(Dispatchers.IO) {
        var conn: HttpsURLConnection? = null
        try {
            val apiBase = getApiBase(context)
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
                try {
                    showToast(context, "HTTP $code : $errorMsg")
                } catch (_: Exception) {
                }
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
            try {
                showToast(context, e.message ?: "Unknown error")
            } catch (_: Exception) {
            }
            throw e
        }
    }

    data class UploadState(
        val name: String,
        val bytesSent: Long,
        val totalBytes: Long,
        val speedBytesPerSec: Long,
        val etaSeconds: Long,
        val progress: Int
    )

    suspend fun uploadModel(
        context: Context,
        name: String,
        params: String,
        fileUri: Uri,
        onProgress: (UploadState) -> Unit
    ) = withContext(Dispatchers.IO) {
        try {
            val apiBase = getApiBase(context)
            val boundary = "----SkynetBoundary${System.currentTimeMillis()}"
            val url = URL("$apiBase/models/upload")
            val conn = (url.openConnection() as HttpsURLConnection).apply {
                requestMethod = "POST"
                connectTimeout = 15000
                readTimeout = 60000
                doOutput = true
                doInput = true
                useCaches = false
                setRequestProperty("Content-Type", "multipart/form-data; boundary=$boundary")
                setChunkedStreamingMode(64 * 1024)
            }

            val displayName = resolveDisplayName(context, fileUri) ?: "model.bin"
            val totalBytes = resolveSize(context, fileUri) ?: -1L

            val prefixName = "--$boundary\r\nContent-Disposition: form-data; name=\"name\"\r\n\r\n"
            val suffix = "\r\n"
            val prefixParams =
                "--$boundary\r\nContent-Disposition: form-data; name=\"params\"\r\n\r\n"
            val prefixFile = buildString {
                append("--$boundary\r\n")
                append("Content-Disposition: form-data; name=\"file\"; filename=\"")
                append(displayName)
                append("\"\r\n")
                append("Content-Type: application/octet-stream\r\n\r\n")
            }
            val endBoundary = "\r\n--$boundary--\r\n"

            var sent = 0L
            var lastTick = System.nanoTime()
            val start = System.nanoTime()

            conn.outputStream.use { os ->
                os.write(prefixName.toByteArray())
                os.write(name.toByteArray())
                os.write(suffix.toByteArray())

                os.write(prefixParams.toByteArray())
                os.write(params.toByteArray())
                os.write(suffix.toByteArray())

                os.write(prefixFile.toByteArray())

                context.contentResolver.openInputStream(fileUri).use { ins ->
                    val buf = ByteArray(DEFAULT_BUFFER_SIZE)
                    while (true) {
                        if (!coroutineContext.isActive) throw CancellationException("Upload cancelled")
                        val read = ins?.read(buf) ?: -1
                        if (read == -1) break
                        os.write(buf, 0, read)
                        sent += read

                        val now = System.nanoTime()
                        val elapsedSec = ((now - start) / 1_000_000_000.0).coerceAtLeast(0.001)
                        val tickMs = (now - lastTick) / 1_000_000
                        if (tickMs >= 500L) {
                            val speed = (sent / elapsedSec).toLong()
                            val remaining =
                                if (totalBytes > 0 && sent <= totalBytes) totalBytes - sent else -1L
                            val eta = if (speed > 0 && remaining > 0) (remaining / speed) else -1L
                            val progress =
                                if (totalBytes > 0) ((sent * 100) / totalBytes).toInt() else 0
                            onProgress(
                                UploadState(
                                    name = name,
                                    bytesSent = sent,
                                    totalBytes = totalBytes,
                                    speedBytesPerSec = speed,
                                    etaSeconds = eta,
                                    progress = progress.coerceIn(0, 100)
                                )
                            )
                            lastTick = now
                        }
                    }
                }

                os.write(endBoundary.toByteArray())
                os.flush()
            }

            val code = conn.responseCode
            if (code !in 200..299) {
                val msg =
                    conn.errorStream?.bufferedReader()?.use { it.readText() } ?: "Upload failed"
                try {
                    showToast(context, "HTTP $code : $msg")
                } catch (_: Exception) {
                }
                throw IllegalStateException("Upload failed with HTTP $code: $msg")
            }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Log.e(TAG, "Exception in uploadModel()", e)
            try {
                showToast(context, e.message ?: "Unknown error")
            } catch (_: Exception) {
            }
            throw e
        }
    }

    fun resolveDisplayName(context: Context, uri: Uri): String? {
        var name: String? = null
        val cursor: Cursor? = context.contentResolver.query(uri, null, null, null, null)
        cursor?.use {
            val idx = it.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (idx >= 0 && it.moveToFirst()) name = it.getString(idx)
        }
        return name
    }

    fun resolveSize(context: Context, uri: Uri): Long? {
        var size: Long? = null
        val cursor: Cursor? = context.contentResolver.query(uri, null, null, null, null)
        cursor?.use {
            val idx = it.getColumnIndex(OpenableColumns.SIZE)
            if (idx >= 0 && it.moveToFirst()) size = it.getLong(idx)
        }
        return size
    }
}
