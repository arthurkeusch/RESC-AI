package com.example.anhilyx.resc_ai;

public class ModelDescriptor {
    private final String modelName;
    private final String modelUrl;
    private final String vocabUrl;

    /**
     * ModelDescriptor constructor.
     * @param modelName The name of the model.
     * @param modelUrl The URL of the model.
     * @param vocabUrl The URL of the model's vocabulary.
     */
    public ModelDescriptor(String modelName, String modelUrl, String vocabUrl) {
        this.modelName = modelName;
        this.modelUrl = modelUrl;
        this.vocabUrl = vocabUrl;
    }

    /**
     * Get the name of the model.
     * @return The name of the model.
     */
    public String getName() {
        return modelName;
    }

    /**
     * Get the URL of the model.
     * @return The URL of the model.
     */
    public String getModelUrl() {
        return modelUrl;
    }

    /**
     * Get the URL of the model's vocabulary.
     * @return The URL of the model's vocabulary.
     */
    public String getVocabUrl() {
        return vocabUrl;
    }

    @Override
    public String toString() {
        return modelName;
    }
}
