package com.example.anhilyx.rescai.benchmark;

import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import com.example.anhilyx.rescai.rag.RAG;
import com.google.android.material.progressindicator.LinearProgressIndicator;

import org.json.JSONException;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.net.URLConnection;
import java.util.concurrent.CompletableFuture;

public class BenchmarkActivity extends AppCompatActivity {

    private byte[] getPDFFromURL(String url) throws IOException {

        // Connect to the URL
        URLConnection connection = new URL(url).openConnection();
        connection.connect();

        // Extract the bytes from the connection
        InputStream is = connection.getInputStream();
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte[] buffer = new byte[8192];
        int bytesRead;
        while ((bytesRead = is.read(buffer)) != -1) {
            baos.write(buffer, 0, bytesRead);
        }

        return baos.toByteArray();
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_benchmark);

        // Retrieve the TextViews and ProgressBars from the layout
        TextView modelName = findViewById(R.id.model_name);
        LinearProgressIndicator modelProgress = findViewById(R.id.model_progress);
        TextView documentName = findViewById(R.id.document_name);
        LinearProgressIndicator documentProgress = findViewById(R.id.document_progress);
        TextView promptName = findViewById(R.id.prompt_name);
        LinearProgressIndicator promptProgress = findViewById(R.id.prompt_progress);
        TextView stepName = findViewById(R.id.step_name);
        LinearProgressIndicator stepProgress = findViewById(R.id.step_progress);

        // Define the documents and prompts for the benchmark
        CompletableFuture.runAsync(() -> {
            Benchmark.DocumentBenchmark[] documents;
            try {
                documents = new Benchmark.DocumentBenchmark[]{
                        new Benchmark.DocumentBenchmark(
                                "La Constitution de la République Française",
                                getPDFFromURL("https://www.conseil-constitutionnel.fr/sites/default/files/as/root/bank_mm/constitution/constitution.pdf"),
                                new Benchmark.PromptBenchmark[]{
                                        new Benchmark.PromptBenchmark("Qui remplace le Président de la République en cas de vacance du pouvoir pour quelque cause que ce soit ?"),
                                        new Benchmark.PromptBenchmark("Quelles sont les conditions strictes pour qu'une révision de la Constitution soit approuvée par référendum ?"),
                                        new Benchmark.PromptBenchmark("Que dit très exactement l'article 49 alinéa 3 concernant l'adoption d'un texte sans vote ?"),
                                        new Benchmark.PromptBenchmark("Quel est le rôle du Conseil constitutionnel dans le contrôle de la validité des lois ?"),
                                        new Benchmark.PromptBenchmark("Selon l'article 1er, quelles sont les caractéristiques fondamentales de la République française ?"),
                                        new Benchmark.PromptBenchmark("Dans quels cas précis et sous quelles conditions le Président peut-il dissoudre l'Assemblée nationale ?"),
                                        new Benchmark.PromptBenchmark("Comment se déroule l'initiative et la promulgation des lois selon le texte constitutionnel ?")
                                }
                        ),
                        new Benchmark.DocumentBenchmark(
                                "La discrimination en intelligence artificielle est-elle suffisamment encadrée ?",
                                getPDFFromURL("https://hal.science/hal-03736828/document"),
                                new Benchmark.PromptBenchmark[]{
                                        new Benchmark.PromptBenchmark("Comment les auteurs de cet article définissent-ils précisément un \"biais algorithmique\" ?"),
                                        new Benchmark.PromptBenchmark("Comment le RGPD protège-t-il les individus contre les décisions de recrutement entièrement automatisées ?"),
                                        new Benchmark.PromptBenchmark("Selon l'étude, est-ce que l'anonymisation des données suffit à éviter les biais discriminatoires ?"),
                                        new Benchmark.PromptBenchmark("Que propose l'article comme solution juridique ou technique pour mieux encadrer le développement de l'IA ?"),
                                        new Benchmark.PromptBenchmark("Quelles limites du droit anti-discrimination actuel sont soulignées dans ce document face aux nouvelles technologies ?"),
                                        new Benchmark.PromptBenchmark("Quel rôle joue l'actuel \"IA Act\" européen dans la mitigation des risques de discrimination évoqués ?")
                                }
                        ),
                        new Benchmark.DocumentBenchmark(
                                "Enseigner et apprendre à l'ère de l'intelligence artificielle",
                                getPDFFromURL("https://hal.science/hal-04013223/document"),
                                new Benchmark.PromptBenchmark[]{
                                        new Benchmark.PromptBenchmark("Quels sont les avantages concrets de l'intelligence artificielle générative pour la préparation des cours par les enseignants ?"),
                                        new Benchmark.PromptBenchmark("Quelles sont les principales inquiétudes soulevées par les auteurs concernant l'évaluation et la notation des élèves à l'ère de l'IA ?"),
                                        new Benchmark.PromptBenchmark("Que recommandent les auteurs concernant l'interdiction d'outils comme ChatGPT à l'école ? Sont-ils pour ou contre ?"),
                                        new Benchmark.PromptBenchmark("Comment l'intelligence artificielle peut-elle favoriser un apprentissage \"personnalisé\" pour l'étudiant selon ce document ?"),
                                        new Benchmark.PromptBenchmark("Quelles compétences spécifiques les étudiants doivent-ils développer aujourd'hui pour utiliser l'IA de manière critique et responsable ?"),
                                        new Benchmark.PromptBenchmark("De manière générale, comment le rôle de l'enseignant est-il amené à évoluer face à ces nouvelles technologies ?")
                                }
                        ),
                        new Benchmark.DocumentBenchmark(
                                "Guide de la sécurité des données personnelles (Édition 2024)",
                                getPDFFromURL("https://www.cnil.fr/sites/cnil/files/2024-03/cnil_guide_securite_personnelle_2024.pdf"),
                                new Benchmark.PromptBenchmark[]{
                                        new Benchmark.PromptBenchmark("Selon les recommandations de la CNIL en 2024, quelles sont les caractéristiques exactes d'un mot de passe considéré comme robuste ?"),
                                        new Benchmark.PromptBenchmark("Quelles sont les étapes obligatoires à suivre en cas de violation de données personnelles au sein d'une entreprise ?"),
                                        new Benchmark.PromptBenchmark("Que dit la fiche dédiée concernant la différence entre la pseudonymisation et l'anonymisation des données ?"),
                                        new Benchmark.PromptBenchmark("Quelle est la durée de conservation maximale recommandée pour les journaux d'accès (logs) informatiques ?"),
                                        new Benchmark.PromptBenchmark("Quelles mesures spécifiques doivent être mises en place pour sécuriser le télétravail des employés ?"),
                                        new Benchmark.PromptBenchmark("Quelles sont les recommandations de la CNIL concernant la destruction ou le recyclage du matériel informatique en fin de vie ?"),
                                        new Benchmark.PromptBenchmark("Quelles clauses de sécurité doivent impérativement figurer dans les contrats lors d'échanges de données avec des sous-traitants ?")
                                }
                        )
                };
            } catch (IOException e) {
                runOnUiThread(() -> {
                    throw new RuntimeException(e);
                });
                throw new RuntimeException(e);
            }

            // Define the models for the benchmark
            Benchmark.ModelBenchmark[] models = new Benchmark.ModelBenchmark[]{
                    new Benchmark.ModelBenchmark("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 128, documents),
                    new Benchmark.ModelBenchmark("intfloat/multilingual-e5-small",                              512, documents),
                    new Benchmark.ModelBenchmark("ibm-granite/granite-embedding-97m-multilingual-r2",           512, documents),
                    new Benchmark.ModelBenchmark("sentence-transformers/all-MiniLM-L6-v2",                      128, documents),
                    new Benchmark.ModelBenchmark("Lajavaness/bilingual-embedding-small",                        1,   documents)
            };

            // Run the benchmark and update the UI with progress
            new Benchmark(models)
                    .run(BenchmarkActivity.this, new Benchmark.BenchmarkProgressCallback() {
                        @Override
                        public void onProgress(float progress, int step, Benchmark.BenchmarkProgress benchmarkProgress) {
                            runOnUiThread(() -> {

                                // Update the model, document, and prompt progress bars and names
                                modelName.setText(benchmarkProgress.currentModel);
                                modelProgress.setProgress((int) (1000.0f * benchmarkProgress.modelsProgress), true);
                                documentName.setText(benchmarkProgress.currentDocument);
                                documentProgress.setProgress((int) (1000.0f * benchmarkProgress.documentsProgress), true);
                                promptName.setText(benchmarkProgress.currentPrompt);
                                promptProgress.setProgress((int) (1000.0f * benchmarkProgress.promptsProgress), true);

                                // Update the step progress bar and name
                                switch (step) {
                                    case RAG.RAGCallback.STEP_INITIALIZING:
                                        stepName.setText("Initializing...");
                                        break;
                                    case RAG.RAGCallback.STEP_INSTALLING_MODEL:
                                        stepName.setText("Downloading model...");
                                        break;
                                    case RAG.RAGCallback.STEP_READING_PDF:
                                        stepName.setText("Reading document...");
                                        break;
                                    case RAG.RAGCallback.STEP_CHUNKING_PDF:
                                        stepName.setText("Chunking document...");
                                        break;
                                    case RAG.RAGCallback.STEP_CREATING_RAG:
                                        stepName.setText("Creating RAG index...");
                                        break;
                                    case RAG.RAGCallback.STEP_QUERYING_RAG:
                                        stepName.setText("Querying RAG...");
                                        break;
                                }
                                stepProgress.setProgress((int) (1000.0f * progress), true);
                            });
                        }
                    })
                    .thenAcceptAsync(results -> {
                        // Reset the UI after the benchmark is complete
                        runOnUiThread(() -> {
                            modelName.setText("");
                            modelProgress.setProgress(0, false);
                            documentName.setText("");
                            documentProgress.setProgress(0, false);
                            promptName.setText("");
                            promptProgress.setProgress(0, false);
                            stepName.setText("Done!");
                            stepProgress.setProgress(100, false);
                        });

                        // Prepare to save the results to a file in the public "Download" folder
                        android.content.ContentValues values = new android.content.ContentValues();
                        values.put(android.provider.MediaStore.MediaColumns.DISPLAY_NAME, "benchmark_results.json");
                        values.put(android.provider.MediaStore.MediaColumns.MIME_TYPE, "application/json");
                        values.put(android.provider.MediaStore.MediaColumns.RELATIVE_PATH, android.os.Environment.DIRECTORY_DOWNLOADS);
                        android.content.ContentResolver resolver = getContentResolver();
                        android.net.Uri uri = resolver.insert(android.provider.MediaStore.Downloads.EXTERNAL_CONTENT_URI, values);

                        // Write the results to the file in the public "Download" folder
                        if (uri != null) {
                            try (java.io.OutputStream os = resolver.openOutputStream(uri)) {
                                if (os != null) {
                                    os.write(results.toString(4).getBytes(java.nio.charset.StandardCharsets.UTF_8));

                                    // Open the results file
                                    runOnUiThread(() -> {
                                        android.content.Intent intent = new android.content.Intent(android.content.Intent.ACTION_VIEW);
                                        intent.setDataAndType(uri, "application/json");
                                        intent.addFlags(android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION);
                                        intent.addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK);
                                        try {
                                            startActivity(intent);
                                        } catch (Exception e) {
                                            Log.e("BENCHMARK_ACTIVITY", "", e);
                                        }
                                    });
                                }
                            } catch (IOException | JSONException e) {
                                Log.e("BENCHMARK_ACTIVITY", "", e);
                            }
                        }

                    })
                    .exceptionally(e -> {
                        Log.e("BENCHMARK_ACTIVITY", "", e);

                        runOnUiThread(() -> {
                            throw new RuntimeException(e);
                        });
                        throw new RuntimeException(e);
                    });
        });
    }
}