package resc.ai.skynetmonitor.config

import android.content.Context
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map

object AppConfig {
    private val Context.dataStore by preferencesDataStore(name = "settings")
    private val KEY_API_URL = stringPreferencesKey("api_url")

    private val _apiUrl = MutableStateFlow("https://resc-ai.arthur-keusch.fr:3000")
    val apiUrl: Flow<String> get() = _apiUrl

    suspend fun init(context: Context) {
        val value = context.dataStore.data.map { it[KEY_API_URL] ?: _apiUrl.value }.first()
        _apiUrl.value = value
    }

    suspend fun setApiUrl(context: Context, url: String) {
        _apiUrl.value = url
        context.dataStore.edit { it[KEY_API_URL] = url }
    }
}
