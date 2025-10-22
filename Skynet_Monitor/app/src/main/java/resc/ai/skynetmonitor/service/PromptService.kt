package resc.ai.skynetmonitor.service

import android.content.Context
import android.net.Uri
import android.util.Log
import android.widget.Toast
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.withContext
import org.json.JSONArray
import resc.ai.skynetmonitor.config.AppConfig
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.HttpURLConnection
import java.net.URL
import javax.net.ssl.HttpsURLConnection

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
                if (!createPrompt(context, apiBase, promptText, datasetId)) {
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
            val payload =
                """{"name":"$name","description":"$description","is_conversational":$isConversational}""".toByteArray()

            val conn = url.openConnection() as HttpsURLConnection
            conn.requestMethod = "POST"
            conn.setRequestProperty("Content-Type", "application/json")
            conn.doOutput = true
            conn.outputStream.write(payload)

            if (conn.responseCode == HttpURLConnection.HTTP_OK) {
                val response = conn.inputStream.bufferedReader().use { it.readText() }
                val match = Regex("\"id_datatset\":(\\d+)").find(response)
                match?.groupValues?.get(1)?.toInt()
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

    private fun createPrompt(
        context: Context,
        apiBase: String,
        prompt: String,
        datasetId: Int
    ): Boolean {
        return try {
            val url = URL("$apiBase/prompts")
            val payload =
                """{"prompt":"$prompt","id_datatset":$datasetId}""".toByteArray()

            val conn = url.openConnection() as HttpsURLConnection
            conn.requestMethod = "POST"
            conn.setRequestProperty("Content-Type", "application/json")
            conn.doOutput = true
            conn.outputStream.write(payload)

            conn.responseCode == HttpURLConnection.HTTP_OK
        } catch (e: Exception) {
            errorToast(context, "Prompt error: ${e.message}")
            false
        }
    }

    suspend fun fetchDatasets(context: Context): List<DatasetItem>? = withContext(Dispatchers.IO) {
        try {
            val apiBase = getApiBase(context)
            val url = URL("$apiBase/datasets")
            val conn = url.openConnection() as HttpsURLConnection
            conn.requestMethod = "GET"

            if (conn.responseCode == HttpURLConnection.HTTP_OK) {
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

    suspend fun deleteDataset(context: Context, datasetId: Int): Boolean = withContext(Dispatchers.IO) {
        try {
            val apiBase = getApiBase(context)
            val url = URL("$apiBase/datasets/$datasetId")
            val conn = url.openConnection() as HttpsURLConnection
            conn.requestMethod = "DELETE"
            conn.setRequestProperty("Content-Type", "application/json")
            conn.connectTimeout = 10000
            conn.readTimeout = 15000

            val responseCode = conn.responseCode
            responseCode in 200..299
        } catch (e: Exception) {
            Log.e(TAG, "Error deleting dataset", e)
            false
        }
    }
}
