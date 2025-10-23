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

object PromptService {
    private const val TAG = "PromptService"

    private fun errorToast(context: Context, message: String) {
        Toast.makeText(context, message, Toast.LENGTH_LONG).show()
    }

    private suspend fun getApiBase(context: Context): String {
        return AppConfig.apiUrl.first()
    }

    suspend fun importDataset(
        context: Context,
        name: String,
        description: String,
        isConversational: Boolean,
        jsonUri: Uri
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

            val datasetId = createDataset(context, apiBase, name, description, isConversational)
                ?: return@withContext false

            for (i in 0 until jsonArray.length()) {
                val obj = jsonArray.getJSONObject(i)
                val promptText = obj.getString("prompt")
                if (!createPromptInternal(context, apiBase, promptText, datasetId)) {
                    errorToast(context, "Failed to create prompt at index $i")
                    return@withContext false
                }
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
}
