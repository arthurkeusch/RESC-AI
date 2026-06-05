package resc.ai.skynetmonitor.service

import android.content.Context
import android.net.Uri
import android.util.Log
import android.widget.Toast
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import resc.ai.skynetmonitor.config.AppConfig
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.HttpURLConnection
import java.net.URL

data class PromptItem(
    val id: Int,
    val prompt: String
)

data class DatasetItem(
    val id: Int,
    val name: String,
    val description: String?,
    val isConversational: Boolean,
    val prompts: List<PromptItem>
)

data class PromptResult(
    val id: Int,
    val response: String,
    val isThink: Boolean,
    val responseTimeMs: Long?,
    val responseTokenCount: Int?,
    val responseTokensPerS: Float?,
    val idPrompt: Int,
    val idModel: Long,
    val idDevices: Int
)

data class DeviceItem(
    val id: Int,
    val name: String
)

data class ModelItem(
    val id: Long,
    val name: String
)

object PromptService {
    private const val TAG = "API_Skynet"

    private fun errorToast(context: Context, message: String) {
        try {
            Toast.makeText(context, message, Toast.LENGTH_LONG).show()
        } catch (_: Exception) {}
    }

    private suspend fun getApiBase(context: Context): String {
        return AppConfig.apiUrl.first()
    }

    suspend fun fetchDevices(context: Context): List<DeviceItem>? = withContext(Dispatchers.IO) {
        try {
            val apiBase = getApiBase(context)
            val url = URL("$apiBase/devices")
            val conn = (url.openConnection() as HttpURLConnection).apply {
                requestMethod = "GET"
            }
            if (conn.responseCode in 200..299) {
                val response = conn.inputStream.bufferedReader().use { it.readText() }
                val arr = JSONArray(response)
                List(arr.length()) { i ->
                    val obj = arr.getJSONObject(i)
                    DeviceItem(obj.getInt("id_devices"), obj.getString("name"))
                }
            } else null
        } catch (e: Exception) {
            Log.e(TAG, "fetchDevices failed", e)
            null
        }
    }

    suspend fun fetchModels(context: Context): List<ModelItem>? = withContext(Dispatchers.IO) {
        try {
            val apiBase = getApiBase(context)
            val url = URL("$apiBase/models")
            val conn = (url.openConnection() as HttpURLConnection).apply {
                requestMethod = "GET"
            }
            if (conn.responseCode in 200..299) {
                val response = conn.inputStream.bufferedReader().use { it.readText() }
                val arr = JSONArray(response)
                List(arr.length()) { i ->
                    val obj = arr.getJSONObject(i)
                    ModelItem(obj.getLong("id_model"), obj.getString("name"))
                }
            } else null
        } catch (e: Exception) {
            Log.e(TAG, "fetchModels failed", e)
            null
        }
    }

    suspend fun fetchResultsForPrompt(context: Context, promptId: Int): List<PromptResult>? = withContext(Dispatchers.IO) {
        try {
            val apiBase = getApiBase(context)
            val url = URL("$apiBase/prompts/$promptId/results")
            val conn = (url.openConnection() as HttpURLConnection).apply {
                requestMethod = "GET"
            }
            if (conn.responseCode in 200..299) {
                val response = conn.inputStream.bufferedReader().use { it.readText() }
                val arr = JSONArray(response)
                List(arr.length()) { i ->
                    val obj = arr.getJSONObject(i)
                    PromptResult(
                        id = obj.getInt("id_result"),
                        response = obj.getString("response"),
                        isThink = obj.optInt("is_think", 0) == 1,
                        responseTimeMs = obj.optLong("response_time_ms").takeIf { it > 0 },
                        responseTokenCount = obj.optInt("response_token_count").takeIf { it > 0 },
                        responseTokensPerS = obj.optDouble("response_tokens_per_s").toFloat().takeIf { it > 0 },
                        idPrompt = obj.getInt("id_prompt"),
                        idModel = obj.getLong("id_model"),
                        idDevices = obj.getInt("id_devices")
                    )
                }
            } else null
        } catch (e: Exception) {
            Log.e(TAG, "fetchResultsForPrompt failed", e)
            null
        }
    }

    suspend fun fetchAllResults(context: Context): List<PromptResult>? = withContext(Dispatchers.IO) {
        try {
            val apiBase = getApiBase(context)
            val url = URL("$apiBase/prompts/results")
            val conn = (url.openConnection() as HttpURLConnection).apply {
                requestMethod = "GET"
            }
            if (conn.responseCode in 200..299) {
                val response = conn.inputStream.bufferedReader().use { it.readText() }
                val arr = JSONArray(response)
                List(arr.length()) { i ->
                    val obj = arr.getJSONObject(i)
                    PromptResult(
                        id = obj.getInt("id_result"),
                        response = obj.getString("response"),
                        isThink = obj.optInt("is_think", 0) == 1,
                        responseTimeMs = obj.optLong("response_time_ms").takeIf { it > 0 },
                        responseTokenCount = obj.optInt("response_token_count").takeIf { it > 0 },
                        responseTokensPerS = obj.optDouble("response_tokens_per_s").toFloat().takeIf { it > 0 },
                        idPrompt = obj.getInt("id_prompt"),
                        idModel = obj.getLong("id_model"),
                        idDevices = obj.getInt("id_devices")
                    )
                }
            } else null
        } catch (e: Exception) {
            Log.e(TAG, "fetchAllResults failed", e)
            null
        }
    }

    suspend fun fetchPerformanceSamples(context: Context, resultId: Int): List<PerformanceSample>? = withContext(Dispatchers.IO) {
        try {
            val apiBase = getApiBase(context)
            val url = URL("$apiBase/prompts/results/$resultId/performance")
            val conn = (url.openConnection() as HttpURLConnection).apply {
                requestMethod = "GET"
            }
            if (conn.responseCode in 200..299) {
                val response = conn.inputStream.bufferedReader().use { it.readText() }
                val arr = JSONArray(response)
                List(arr.length()) { i ->
                    val obj = arr.getJSONObject(i)
                    PerformanceSample(
                        sampleTimeMs = obj.getLong("sample_time_ms"),
                        batteryPercent = obj.optDouble("battery_percent").toFloat(),
                        ramCurrentMb = obj.optDouble("ram_current_mb").toFloat(),
                        ramMaxMb = obj.optDouble("ram_max_mb").toFloat(),
                        batteryTemperatureC = obj.optDouble("battery_temperature_c").toFloat()
                    )
                }
            } else null
        } catch (e: Exception) {
            Log.e(TAG, "fetchPerformanceSamples failed", e)
            null
        }
    }

    suspend fun importDataset(
        context: Context,
        name: String,
        description: String,
        isConversational: Boolean,
        jsonUri: Uri,
        onProgress: (Int, Int) -> Unit = { _, _ -> },
        onDatasetCreated: (Int) -> Unit = {}
    ): Boolean = withContext(Dispatchers.IO) {
        try {
            val apiBase = getApiBase(context)
            val inputStream = context.contentResolver.openInputStream(jsonUri)
                ?: run {
                    errorToast(context, "Failed to open JSON file")
                    return@withContext false
                }
            val reader = BufferedReader(InputStreamReader(inputStream))
            val jsonText = reader.use { it.readText() }
            val jsonArray = JSONArray(jsonText)

            val promptsToCreate = mutableListOf<String>()
            for (i in 0 until jsonArray.length()) {
                val obj = jsonArray.getJSONObject(i)
                if (obj.has("prompt")) {
                    promptsToCreate.add(obj.getString("prompt"))
                }
            }

            if (promptsToCreate.isEmpty()) {
                errorToast(context, "No prompts found in file")
                return@withContext false
            }

            val datasetId = createDataset(context, apiBase, name, description, isConversational)
                ?: return@withContext false
            
            onDatasetCreated(datasetId)

            onProgress(0, promptsToCreate.size)
            for (i in promptsToCreate.indices) {
                if (!createPromptInternal(context, apiBase, promptsToCreate[i], datasetId)) {
                    errorToast(context, "Failed to create prompt at index $i")
                    return@withContext false
                }
                onProgress(i + 1, promptsToCreate.size)
            }
            true
        } catch (e: Exception) {
            Log.e(TAG, "Error during dataset import", e)
            errorToast(context, "Error during import: ${e.message}")
            false
        }
    }

    private fun createDataset(
        context: Context,
        apiBase: String,
        name: String,
        description: String,
        isConversational: Boolean
    ): Int? {
        return try {
            val url = URL("$apiBase/datasets")
            val payload = JSONObject().apply {
                put("name", name)
                put("description", description)
                put("is_conversational", if (isConversational) 1 else 0)
            }.toString().toByteArray()

            val conn = (url.openConnection() as HttpURLConnection).apply {
                requestMethod = "POST"
                setRequestProperty("Content-Type", "application/json")
                doOutput = true
                connectTimeout = 10000
                readTimeout = 15000
                outputStream.use { it.write(payload) }
            }

            if (conn.responseCode in 200..299) {
                val response = conn.inputStream.bufferedReader().use { it.readText() }
                val obj = JSONObject(response)
                obj.optInt("id_datatset", -1).takeIf { it >= 0 }
            } else {
                val errorMsg = conn.errorStream?.bufferedReader()?.use { it.readText() }
                errorToast(context, "Dataset creation failed: $errorMsg")
                null
            }
        } catch (e: Exception) {
            errorToast(context, "Dataset error: ${e.message}")
            null
        }
    }

    private fun createPromptInternal(
        context: Context,
        apiBase: String,
        prompt: String,
        datasetId: Int
    ): Boolean {
        return try {
            val url = URL("$apiBase/prompts")
            val payload = JSONObject().apply {
                put("prompt", prompt)
                put("id_datatset", datasetId)
            }.toString().toByteArray()

            val conn = (url.openConnection() as HttpURLConnection).apply {
                requestMethod = "POST"
                setRequestProperty("Content-Type", "application/json")
                doOutput = true
                connectTimeout = 10000
                readTimeout = 15000
                outputStream.use { it.write(payload) }
            }

            conn.responseCode in 200..299
        } catch (e: Exception) {
            errorToast(context, "Prompt error: ${e.message}")
            false
        }
    }

    suspend fun addPrompt(
        context: Context,
        datasetId: Int,
        promptText: String
    ): Boolean = withContext(Dispatchers.IO) {
        try {
            val apiBase = getApiBase(context)
            createPromptInternal(context, apiBase, promptText, datasetId)
        } catch (e: Exception) {
            errorToast(context, "Add prompt error: ${e.message}")
            false
        }
    }

    suspend fun fetchDatasets(context: Context): List<DatasetItem>? = withContext(Dispatchers.IO) {
        try {
            val apiBase = getApiBase(context)
            val url = URL("$apiBase/datasets")
            val conn = (url.openConnection() as HttpURLConnection).apply {
                requestMethod = "GET"
                connectTimeout = 10000
                readTimeout = 15000
            }

            if (conn.responseCode in 200..299) {
                val response = conn.inputStream.bufferedReader().use { it.readText() }
                val jsonArray = JSONArray(response)
                val output = mutableListOf<DatasetItem>()

                for (i in 0 until jsonArray.length()) {
                    val obj = jsonArray.getJSONObject(i)

                    val promptsJson = obj.getJSONArray("prompts")
                    val prompts = mutableListOf<PromptItem>()
                    for (j in 0 until promptsJson.length()) {
                        val p = promptsJson.getJSONObject(j)
                        prompts.add(
                            PromptItem(
                                id = p.getInt("id_prompt"),
                                prompt = p.getString("prompt")
                            )
                        )
                    }

                    output.add(
                        DatasetItem(
                            id = obj.getInt("id_datatset"),
                            name = obj.getString("name"),
                            description = obj.optString("description", null),
                            isConversational = obj.optInt("is_conversational", 0) == 1,
                            prompts = prompts
                        )
                    )
                }
                output
            } else {
                errorToast(context, "Failed to load datasets")
                null
            }
        } catch (e: Exception) {
            errorToast(context, "Error: ${e.message}")
            null
        }
    }

    suspend fun updateDataset(
        context: Context,
        datasetId: Int,
        name: String,
        description: String?,
        isConversational: Boolean
    ): Boolean = withContext(Dispatchers.IO) {
        try {
            val apiBase = getApiBase(context)
            val url = URL("$apiBase/datasets/$datasetId")
            val payload = JSONObject().apply {
                put("name", name)
                put("description", description ?: JSONObject.NULL)
                put("is_conversational", if (isConversational) 1 else 0)
            }.toString().toByteArray()

            val conn = (url.openConnection() as HttpURLConnection).apply {
                requestMethod = "PUT"
                setRequestProperty("Content-Type", "application/json")
                doOutput = true
                connectTimeout = 10000
                readTimeout = 15000
                outputStream.use { it.write(payload) }
            }

            conn.responseCode in 200..299
        } catch (e: Exception) {
            errorToast(context, "Update dataset error: ${e.message}")
            false
        }
    }

    suspend fun updatePrompt(
        context: Context,
        promptId: Int,
        newText: String,
        datasetId: Int
    ): Boolean = withContext(Dispatchers.IO) {
        try {
            val apiBase = getApiBase(context)
            val url = URL("$apiBase/prompts/$promptId")
            val payload = JSONObject().apply {
                put("prompt", newText)
                put("id_datatset", datasetId)
            }.toString().toByteArray()

            val conn = (url.openConnection() as HttpURLConnection).apply {
                requestMethod = "PUT"
                setRequestProperty("Content-Type", "application/json")
                doOutput = true
                connectTimeout = 10000
                readTimeout = 15000
                outputStream.use { it.write(payload) }
            }

            conn.responseCode in 200..299
        } catch (e: Exception) {
            errorToast(context, "Update prompt error: ${e.message}")
            false
        }
    }

    suspend fun deleteDataset(context: Context, datasetId: Int): Boolean =
        withContext(Dispatchers.IO) {
            try {
                val apiBase = getApiBase(context)
                val url = URL("$apiBase/datasets/$datasetId")
                val conn = (url.openConnection() as HttpURLConnection).apply {
                    requestMethod = "DELETE"
                    setRequestProperty("Content-Type", "application/json")
                    connectTimeout = 10000
                    readTimeout = 15000
                }
                conn.responseCode in 200..299
            } catch (e: Exception) {
                Log.e(TAG, "Error deleting dataset", e)
                false
            }
        }

    suspend fun deletePrompt(context: Context, promptId: Int): Boolean =
        withContext(Dispatchers.IO) {
            try {
                val apiBase = getApiBase(context)
                val url = URL("$apiBase/prompts/$promptId")
                val conn = (url.openConnection() as HttpURLConnection).apply {
                    requestMethod = "DELETE"
                    setRequestProperty("Content-Type", "application/json")
                    connectTimeout = 10000
                    readTimeout = 15000
                }
                conn.responseCode in 200..299
            } catch (e: Exception) {
                errorToast(context, "Delete prompt error: ${e.message}")
                false
            }
        }

    suspend fun submitBenchmarkResult(
        context: Context,
        response: String,
        idPrompt: Int,
        idModel: Long,
        idDevices: Int,
        isThink: Boolean,
        responseTimeMs: Long,
        responseTokenCount: Int,
        responseTokensPerS: Float,
        performanceSamples: List<PerformanceSample>
    ): Boolean = withContext(Dispatchers.IO) {
        try {
            val apiBase = getApiBase(context)
            val url = URL("$apiBase/prompts/results")
            val payloadObj = JSONObject().apply {
                put("response", response)
                put("id_prompt", idPrompt)
                put("id_model", idModel)
                put("id_devices", idDevices)
                put("is_think", isThink)
                put("response_time_ms", responseTimeMs)
                put("response_token_count", responseTokenCount)
                put("response_tokens_per_s", responseTokensPerS)
                
                val samplesArray = JSONArray()
                performanceSamples.forEach { s ->
                    samplesArray.put(JSONObject().apply {
                        put("sample_time_ms", s.sampleTimeMs)
                        put("battery_percent", s.batteryPercent)
                        put("ram_current_mb", s.ramCurrentMb)
                        put("ram_max_mb", s.ramMaxMb)
                        put("battery_temperature_c", s.batteryTemperatureC)
                    })
                }
                put("performance_samples", samplesArray)
            }
            val payload = payloadObj.toString()
            Log.d("API_Skynet", "POST $url | Payload size: ${payload.length}")

            val conn = (url.openConnection() as HttpURLConnection).apply {
                requestMethod = "POST"
                setRequestProperty("Content-Type", "application/json")
                doOutput = true
                connectTimeout = 10000
                readTimeout = 15000
                outputStream.use { it.write(payload.toByteArray()) }
            }

            val status = conn.responseCode
            val responseBody = if (status in 200..299) {
                conn.inputStream.bufferedReader().use { it.readText() }
            } else {
                conn.errorStream?.bufferedReader()?.use { it.readText() }
            }
            Log.d("API_Skynet", "Response ($status): $responseBody")

            status in 200..299
        } catch (e: Exception) {
            Log.e("API_Skynet", "Result submission failed", e)
            false
        }
    }

    suspend fun registerOrGetDevice(context: Context, info: Map<String, String>): Int? = withContext(Dispatchers.IO) {
        try {
            val apiBase = getApiBase(context)
            val url = URL("$apiBase/devices")
            val payload = JSONObject().apply {
                put("brand", info["Brand"])
                put("model", info["Model"])
                put("android_version", info["Android Version"])
                put("cpu_arch", info["CPU Architecture"])
                put("cpu_cores", info["CPU Cores"]?.toIntOrNull() ?: 0)
                put("soc", info["SoC"])
                put("gpu", info["GPU"])
                put("ram", info["RAM"])
            }.toString()
            
            Log.d("API_Skynet", "POST $url | Registering device: $payload")

            val conn = (url.openConnection() as HttpURLConnection).apply {
                requestMethod = "POST"
                setRequestProperty("Content-Type", "application/json")
                doOutput = true
                connectTimeout = 10000
                readTimeout = 15000
                outputStream.use { it.write(payload.toByteArray()) }
            }

            val status = conn.responseCode
            if (status in 200..299) {
                val response = conn.inputStream.bufferedReader().use { it.readText() }
                Log.d("API_Skynet", "Device Reg Response ($status): $response")
                val obj = JSONObject(response)
                obj.optInt("id_devices", -1).takeIf { it >= 0 }
            } else {
                val err = conn.errorStream?.bufferedReader()?.use { it.readText() }
                Log.e("API_Skynet", "Device registration failed ($status): $err")
                null
            }
        } catch (e: Exception) {
            Log.e("API_Skynet", "Device registration error", e)
            null
        }
    }
}
