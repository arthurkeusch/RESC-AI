package resc.ai.skynetmonitor.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import resc.ai.skynetmonitor.service.*

data class StatsState(
    val isLoading: Boolean = false,
    val datasets: List<DatasetItem> = emptyList(),
    val devices: List<DeviceItem> = emptyList(),
    val models: List<ModelItem> = emptyList(),
    val selectedDeviceFilter: Int? = null, // null means "All"
    val navigation: StatsNavigation = StatsNavigation.DatasetList
)

sealed class StatsNavigation {
    object DatasetList : StatsNavigation()
    data class DatasetStats(val dataset: DatasetItem) : StatsNavigation()
    data class PromptList(val dataset: DatasetItem) : StatsNavigation()
    data class PromptDetail(val prompt: PromptItem, val dataset: DatasetItem) : StatsNavigation()
    object ModelComparison : StatsNavigation()
}

class StatsViewModel(application: Application) : AndroidViewModel(application) {
    private val _state = MutableStateFlow(StatsState())
    val state: StateFlow<StatsState> = _state.asStateFlow()

    private val ctx = application.applicationContext

    init {
        loadInitialData()
    }

    fun loadInitialData() {
        viewModelScope.launch {
            _state.update { it.copy(isLoading = true) }
            val ds = PromptService.fetchDatasets(ctx) ?: emptyList()
            val dev = PromptService.fetchDevices(ctx) ?: emptyList()
            val mod = PromptService.fetchModels(ctx) ?: emptyList()
            _state.update { it.copy(
                isLoading = false,
                datasets = ds,
                devices = dev,
                models = mod
            ) }
        }
    }

    fun navigateTo(nav: StatsNavigation) {
        _state.update { it.copy(navigation = nav) }
    }

    fun setDeviceFilter(deviceId: Int?) {
        _state.update { it.copy(selectedDeviceFilter = deviceId) }
    }

    suspend fun fetchAllResults(): List<PromptResult> {
        return PromptService.fetchAllResults(ctx) ?: emptyList()
    }

    // Helper to fetch results for a prompt
    suspend fun getResultsForPrompt(promptId: Int): List<PromptResult> {
        return PromptService.fetchResultsForPrompt(ctx, promptId) ?: emptyList()
    }

    // Helper to fetch performance samples for a result
    suspend fun getPerformanceSamples(resultId: Int): List<PerformanceSample> {
        return PromptService.fetchPerformanceSamples(ctx, resultId) ?: emptyList()
    }
}
