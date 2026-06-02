package com.example.anhilyx.resc_ai;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class RagExtractionFragment extends Fragment {

    private static final int N_RESULTS = 5;

    private EditText textQuery;
    private TextView result;
    private TextView pageIdx;
    private Button btnPrev;
    private Button btnNext;

    private RagManager ragManager;
    private EmbeddingEngine embeddingEngine;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();

    private final List<String> retrievedTexts = new ArrayList<>();
    private int currentChunkIndex = 0;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {

        // Create the view
        View view = inflater.inflate(R.layout.fragment_rag_extraction, container, false);

        // Initialize the objects
        ragManager = new RagManager();
        embeddingEngine = new EmbeddingEngine(requireContext());

        // Retrieve the elements
        textQuery = view.findViewById(R.id.textQuery);
        result = view.findViewById(R.id.result);
        pageIdx = view.findViewById(R.id.pageidx);
        btnPrev = view.findViewById(R.id.btnPrev);
        btnNext = view.findViewById(R.id.btnNext);
        Button btnSearch = view.findViewById(R.id.btnSearch);

        // Assign listeners
        btnSearch.setOnClickListener(v -> performSearch());
        btnPrev.setOnClickListener(v -> navigateToChunk(-1));
        btnNext.setOnClickListener(v -> navigateToChunk(1));

        return view;
    }

    /**
     * Perform the search.
     */
    private void performSearch() {
        String queryText = textQuery.getText().toString().trim();
        if (queryText.isEmpty()) return;

        executor.execute(() -> {
            float[] queryEmbedding = embeddingEngine.generateEmbedding(queryText);
            List<String> results = ragManager.retrieveChunks(queryEmbedding, N_RESULTS);

            requireActivity().runOnUiThread(() -> {
                retrievedTexts.clear();
                retrievedTexts.addAll(results);
                currentChunkIndex = 0;
                updateUi();
            });
        });
    }

    /**
     * Navigate to the given chunk.
     * @param direction The direction of the navigation.
     */
    private void navigateToChunk(int direction) {
        currentChunkIndex = Math.max(0, Math.min(currentChunkIndex + direction, retrievedTexts.size() - 1));
        updateUi();
    }

    private void updateUi() {

        // On empty results, display a message
        if (retrievedTexts.isEmpty()) {
            result.setText("No results found in the vector index.");
            pageIdx.setText("0 / 0");
            btnPrev.setEnabled(false);
            btnNext.setEnabled(false);
            return;
        }

        // Else, display the chunk
        result.setText(retrievedTexts.get(currentChunkIndex));
        pageIdx.setText((currentChunkIndex + 1) + " / " + retrievedTexts.size());
        btnPrev.setEnabled(currentChunkIndex > 0);
        btnNext.setEnabled(currentChunkIndex < retrievedTexts.size() - 1);
    }
}