package resc.ai.skynetmonitor.service

object NativeLlama {
    init {
        System.loadLibrary("llama")
    }

    external fun initModel(modelPath: String): Boolean
    external fun generate(prompt: String): String?
    external fun freeModel()
}
