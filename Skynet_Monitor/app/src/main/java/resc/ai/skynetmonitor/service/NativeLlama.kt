package resc.ai.skynetmonitor.service

object NativeLlama {

    init {
        System.loadLibrary("llama-android") // lib générée par le CMake de l’exemple
    }

    // --- Variables internes pour stocker le modèle et le contexte ---
    private var modelPtr: Long = 0
    private var contextPtr: Long = 0
    private var batchPtr: Long = 0
    private var samplerPtr: Long = 0

    // === Fonctions publiques pour ton app ===

    /** Charge le modèle depuis un fichier */
    fun initModel(modelPath: String): Boolean {
        // backend init
        backend_init()
        log_to_android()

        modelPtr = load_model(modelPath)
        if (modelPtr == 0L) return false

        contextPtr = new_context(modelPtr)
        batchPtr = new_batch(512, 0, 8) // paramètres par défaut
        samplerPtr = new_sampler()

        return true
    }

    /** Génère du texte à partir du prompt */
    fun generate(prompt: String, maxTokens: Int = 128): String? {
        if (modelPtr == 0L || contextPtr == 0L) return null

        val nCur = IntVar(0)
        completion_init(contextPtr, batchPtr, prompt, false, maxTokens)

        val sb = StringBuilder()
        while (true) {
            val token = completion_loop(contextPtr, batchPtr, samplerPtr, maxTokens, nCur)
            if (token.isNullOrEmpty()) break
            sb.append(token)
        }
        return sb.toString()
    }

    /** Libère le modèle et le contexte de la mémoire */
    fun freeModel() {
        if (samplerPtr != 0L) {
            free_sampler(samplerPtr)
            samplerPtr = 0
        }
        if (batchPtr != 0L) {
            free_batch(batchPtr)
            batchPtr = 0
        }
        if (contextPtr != 0L) {
            free_context(contextPtr)
            contextPtr = 0
        }
        if (modelPtr != 0L) {
            free_model(modelPtr)
            modelPtr = 0
        }
        backend_free()
    }

    // === Fonctions JNI internes liées au C++ ===
    private external fun load_model(modelPath: String): Long
    private external fun free_model(modelPointer: Long)
    private external fun new_context(modelPointer: Long): Long
    private external fun free_context(contextPointer: Long)
    private external fun kv_cache_clear(contextPointer: Long)
    private external fun new_batch(nTokens: Int, embd: Int, nSeqMax: Int): Long
    private external fun free_batch(batchPointer: Long)
    private external fun new_sampler(): Long
    private external fun free_sampler(samplerPointer: Long)
    private external fun backend_init()
    private external fun backend_free()
    private external fun log_to_android()
    private external fun completion_init(
        contextPointer: Long,
        batchPointer: Long,
        text: String,
        formatChat: Boolean,
        nLen: Int
    ): Int
    private external fun completion_loop(
        contextPointer: Long,
        batchPointer: Long,
        samplerPointer: Long,
        nLen: Int,
        nCurVar: IntVar
    ): String?

    /** Petit helper pour passer un int mutable à completion_loop */
    class IntVar(var value: Int) {
        fun inc() { value++ }
    }
}
