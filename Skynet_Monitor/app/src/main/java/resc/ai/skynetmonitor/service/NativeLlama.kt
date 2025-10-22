package resc.ai.skynetmonitor.service

object NativeLlama {
    init {
        System.loadLibrary("llama")
    }
    init {
        System.loadLibrary("native-lib") // doit être appelé avant d'utiliser initModel
    }

    external fun initModel(modelPath: String): Boolean
    external fun generate(prompt: String): String?
    external fun freeModel()
}
