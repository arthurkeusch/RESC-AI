// Kotlin
package resc.ai.skynetmonitor.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import java.util.PriorityQueue
import kotlin.math.sqrt
import resc.ai.skynetmonitor.service.ReadOnlyDatabase
import android.util.Log

@Composable
fun RagScreen(
    innerPadding: PaddingValues
) {
    val context = LocalContext.current

    val vocab = ReadOnlyDatabase(context, "vocab")
    // val rag = ReadOnlyDatabase(context, "lol_champs_rag")
    val rag = ReadOnlyDatabase(context, "jurisprudence_rag")

    // Load model
    /* val modelName = "all-minilm-l12-v2-q4_k_m.gguf"
    val modelFile = File(context.filesDir, modelName)
    if (!modelFile.exists()) {
        try {
            context.assets.open(modelName).use { inputStream ->
                modelFile.outputStream().use { outputStream ->
                    inputStream.copyTo(outputStream)
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
    NativeLlama.initModel(modelFile.absolutePath) */

    var query by remember { mutableStateOf("Prompt") }

    val results = remember { mutableStateListOf<String>() }

    fun generateEmbedding(text: String): FloatArray? {
        // Find each word embedding
        val words =  text.split(" ", "\n", "\t")
        val wordEmbeddings = mutableListOf<FloatArray>()
        for (word in words) {
            val index = vocab.getIndex(word)
            if (index != -1) {
                wordEmbeddings.add(vocab.getEmbeddings(index))
            }
        }
        if (wordEmbeddings.isEmpty()) return null

        // Average word embeddings
        val embedding = FloatArray(wordEmbeddings[0].size) { 0f }
        for (we in wordEmbeddings) {
            for (i in we.indices) {
                embedding[i] += we[i]
            }
        }
        for (i in embedding.indices) {
            embedding[i] /= wordEmbeddings.size
        }
        return embedding
    }

    fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        if (a.size != b.size || a.isEmpty()) return -1f
        var dot = 0f
        var normA = 0f
        var normB = 0f
        for (i in a.indices) {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        return dot / ((sqrt(normA.toDouble()) * sqrt(normB.toDouble())).toFloat() + 1e-10f)
    }

    fun search(query: String, k: Int): MutableList<String> {
        Log.d("ROD", "Search for query: $query")
        // min-heap by similarity (keep top k largest similarities)
        val heap = PriorityQueue<Pair<Int, Float>>(compareBy { it.second })

        val words = query.split(
            " ", "\n", "\t",
            ",", "'", ".", "!", "?", ";", ":",
            "(", ")", "[", "]", "{", "}", "<", ">", "\""
        )
        val queryEmbeddings = mutableListOf<FloatArray>()
        for (word in words) {
            val index = vocab.getIndex(word.lowercase())
            if (index == -1) {
                Log.d("ROD", "Mot inconnu : $word")
            } else {
                Log.d("ROD", "OK : $word")
                queryEmbeddings.add(vocab.getEmbeddings(index))
            }
        }
        val queryEmbedding = mutableListOf<Float>()
        for (i in 0 until vocab.getEmbeddings(0).size) {
            var sum = 0f
            for (qe in queryEmbeddings) {
                sum += qe[i]
            }
            queryEmbedding.add(sum / queryEmbeddings.size)
            // Log.d("ROD", "Embedding dim $i : ${queryEmbedding.last()}")
        }

        for (i in 0 until rag.size()) {
            val sim = cosineSimilarity( queryEmbedding.toFloatArray(), rag.getEmbeddings(i))
            heap.add(Pair(i, sim))
            if (heap.size > k) heap.poll()
            // Log.d("ROD", "RAG doc $i similarity: $sim")
        }

        val temp = mutableListOf<Pair<Int, Float>>()
        while (heap.isNotEmpty()) temp.add(heap.poll())
        temp.sortByDescending { it.second }

        val topK = mutableListOf<String>()
        for (p in temp) {
            Log.d("ROD", "Embedding " + p.first)
            topK.add(rag.getString(p.first))
        }

        Log.d("ROD", "Top $k results: $topK")
        return topK
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(innerPadding)
    ) {
        OutlinedTextField(
            value = query,
            onValueChange = { query = it },
            label = { Text("Prompt") },
            modifier = Modifier
                .fillMaxWidth()
                .padding(8.dp)
        )

        Button(
            onClick = {
                results.clear()
                if (vocab.size() == 0 || rag.size() == 0) {
                    results.add("Aucun fichier chargé !")
                } else {
                    results.addAll(search(query, 5))
                }
            },
            modifier = Modifier
                .fillMaxWidth()
                .padding(8.dp)
        ) {
            Text("Get Texts")
        }

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(8.dp)
                .verticalScroll(rememberScrollState())
        ) {
            results.forEachIndexed { i, text ->
                if (i > 0) {
                    Spacer(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(2.dp)
                            .background(Color.Gray)
                    )
                }
                Text(
                    text = "${i + 1}. $text",
                    modifier = Modifier.padding(vertical = 4.dp)
                )
            }
        }
    }
}
