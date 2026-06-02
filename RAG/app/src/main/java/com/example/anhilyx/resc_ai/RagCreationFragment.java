package com.example.anhilyx.resc_ai;

import android.content.Context;
import android.content.SharedPreferences;
import android.net.Uri;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import com.google.android.material.progressindicator.LinearProgressIndicator;
import com.tom_roush.pdfbox.android.PDFBoxResourceLoader;
import com.tom_roush.pdfbox.pdmodel.PDDocument;
import com.tom_roush.pdfbox.text.PDFTextStripper;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class RagCreationFragment extends Fragment {

    private static final int CHUNK_SIZE = 1024;

    private Spinner modelSpinner;
    private Button pdfButton;
    private LinearProgressIndicator progressIndicator;
    private TextView textStatus;

    private EmbeddingEngine embeddingEngine;
    private RagManager ragManager;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final ActivityResultLauncher<String> pickPdfLauncher = registerForActivityResult(
            new ActivityResultContracts.GetContent(), this::processSelectedPdf
    );

    private final static ModelDescriptor[] MODELS = {
            new ModelDescriptor(
                    "MiniLM-L6",
                    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx?download=true",
                    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt?download=true"
            ),
            new ModelDescriptor(
                    "BGE Small (en)",
                    "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx?download=true",
                    "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/vocab.txt?download=true"
            )
    };

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {

        View view = inflater.inflate(R.layout.fragment_rag_creation, container, false);

        modelSpinner = view.findViewById(R.id.modelSpinner);
        pdfButton = view.findViewById(R.id.btnSelectPdf);
        progressIndicator = view.findViewById(R.id.progressIndicator);
        textStatus = view.findViewById(R.id.tvStatus);

        embeddingEngine = new EmbeddingEngine(requireContext());
        ragManager = new RagManager();

        PDFBoxResourceLoader.init(requireContext());

        ArrayAdapter<ModelDescriptor> adapter = new ArrayAdapter<>(requireContext(), android.R.layout.simple_spinner_dropdown_item, MODELS);
        modelSpinner.setAdapter(adapter);

        pdfButton.setOnClickListener(v -> pickPdfLauncher.launch("application/pdf"));

        return view;
    }

    /**
     * Process the selected PDF file.
     * @param uri The URI of the selected PDF file.
     */
    private void processSelectedPdf(Uri uri) {
        if (uri == null) return;

        progressIndicator.setVisibility(View.VISIBLE);
        pdfButton.setEnabled(false);

        executor.execute(() -> {
            // Download the embedding model (if needed)
            updateStatus("Status: Downloading model...", false);
            embeddingEngine.downloadModel((ModelDescriptor) modelSpinner.getSelectedItem(), new EmbeddingEngine.ModelDownloadCallback() {
                @Override
                public void onSuccess() {
                    // Extract text from the PDF
                    updateStatus("Status: Extracting text from PDF...", false);
                    String pdfText = "";
                    try {
                        InputStream is = requireContext().getContentResolver().openInputStream(uri);
                        PDDocument document = PDDocument.load(is);
                        PDFTextStripper stripper = new PDFTextStripper();
                        pdfText = stripper.getText(document);
                        document.close();
                    } catch (Exception e) {
                        updateStatus("PDF error: " + e.getMessage(), true);
                        return;
                    }

                    updateStatus("Status: Indexing chunks...", false);
                    try {
                        List<String> chunks = splitIntoChunks(pdfText);
                        for (String chunk : chunks) {
                            float[] vector = embeddingEngine.generateEmbedding(chunk);
                            ragManager.ingestDocument(chunk, vector);
                        }
                    } catch (Exception e) {
                        updateStatus("Embedding error: " + e.getMessage(), true);
                        return;
                    }

                    // Update app with success
                    SharedPreferences prefs = requireActivity().getSharedPreferences("rag_prefs", Context.MODE_PRIVATE);
                    prefs.edit()
                            .putBoolean("is_rag_created", true)
                            .apply();
                    updateStatus("Status: Finished!", false);
                    requireActivity().runOnUiThread(() -> {
                        progressIndicator.setVisibility(View.GONE);
                        pdfButton.setEnabled(true);

                        ((MainActivity) requireActivity()).onRagCreated();
                    });

                }

                @Override
                public void onError(String error) {
                    updateStatus(error, true);
                }
            });
        });
    }

    /**
     * Split the given text into chunks of (more or less) the given size.
     * In fact, the chunks have a minimum size of CHUNK_SIZE, but will only end at whitespaces or punctuation.
     * @param text The text to split.
     * @return The list of chunks.
     */
    private List<String> splitIntoChunks(String text) {
        List<String> chunks = new ArrayList<>();
        int i = 0;
        int length = text.length();
        while (i < length) {
            int j = Math.min(i + RagCreationFragment.CHUNK_SIZE, length);
            if (j < i + RagCreationFragment.CHUNK_SIZE) {
                while (
                        j < length &&
                        Character.isLetterOrDigit(text.charAt(j))
                ) { j++; }
            }
            chunks.add(text.substring(i, j));
            i = j;
        }
        return chunks;
    }

    /**
     * Update the status text on the UI thread.
     * @param message The message to display.
     * @param isError True if the message is an error, false otherwise.
     */
    private void updateStatus(String message, boolean isError) {
        requireActivity().runOnUiThread(() -> {
            textStatus.setText(message);
            if (isError) {
                progressIndicator.setVisibility(View.GONE);
                pdfButton.setEnabled(true);
                Toast.makeText(getContext(), message, Toast.LENGTH_LONG).show();
            }
        });
    }
}