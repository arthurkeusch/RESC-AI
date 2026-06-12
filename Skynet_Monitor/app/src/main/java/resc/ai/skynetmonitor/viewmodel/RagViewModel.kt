package resc.ai.skynetmonitor.viewmodel

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.anhilyx.rescai.rag.RAG
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.InputStream

class RagViewModel(application: Application) : AndroidViewModel(application) {

    private val _indexedFiles = MutableStateFlow<List<String>>(emptyList())
    val indexedFiles: StateFlow<List<String>> = _indexedFiles.asStateFlow()

    private val _ragInfo = MutableStateFlow("")
    val ragInfo: StateFlow<String> = _ragInfo.asStateFlow()

    private val _searchResults = MutableStateFlow<List<String>>(emptyList())
    val searchResults: StateFlow<List<String>> = _searchResults.asStateFlow()

    private val _isInitializing = MutableStateFlow(false)
    val isInitializing: StateFlow<Boolean> = _isInitializing.asStateFlow()

    private val _isProcessing = MutableStateFlow(false)
    val isProcessing: StateFlow<Boolean> = _isProcessing.asStateFlow()

    private val _progress = MutableStateFlow(0f)
    val progress: StateFlow<Float> = _progress.asStateFlow()

    init {
        initRag()
    }

    fun initRag() {
        _isInitializing.value = true
        viewModelScope.launch(Dispatchers.IO) {
            RAG.init(getApplication(), object : RAG.RAGCallback {
                override fun onSuccess() {
                    _isInitializing.value = false
                    refreshInfo()
                }

                override fun onError(e: Exception, step: Int) {
                    _isInitializing.value = false
                    // Handle error
                }

                override fun onProgress(progress: Float, step: Int) {
                    _progress.value = progress
                }
            })
        }
    }

    fun refreshInfo() {
        _indexedFiles.value = RAG.getIndexedFiles()
        _ragInfo.value = RAG.getRagInfo()
    }

    fun addFile(uri: Uri, fileName: String) {
        viewModelScope.launch(Dispatchers.IO) {
            _isProcessing.value = true
            val inputStream: InputStream? = getApplication<Application>().contentResolver.openInputStream(uri)
            if (inputStream != null) {
                RAG.createRAG(inputStream, fileName, object : RAG.RAGCallback {
                    override fun onSuccess() {
                        _isProcessing.value = false
                        refreshInfo()
                    }

                    override fun onError(e: Exception, step: Int) {
                        _isProcessing.value = false
                    }

                    override fun onProgress(progress: Float, step: Int) {
                        _progress.value = progress
                    }
                })
            } else {
                _isProcessing.value = false
            }
        }
    }

    fun removeFile(fileName: String) {
        RAG.removeFile(fileName)
        refreshInfo()
    }

    fun clearIndex() {
        RAG.clearIndex()
        refreshInfo()
    }

    fun search(query: String) {
        if (query.isBlank()) {
            _searchResults.value = emptyList()
            return
        }
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val results = RAG.queryRAG(query)
                _searchResults.value = results.toList()
            } catch (e: Exception) {
                _searchResults.value = listOf("Error: ${e.message}")
            }
        }
    }
}
