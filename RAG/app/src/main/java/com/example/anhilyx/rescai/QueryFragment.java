package com.example.anhilyx.rescai;

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

import com.example.anhilyx.rescai.rag.RAG;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class QueryFragment extends Fragment {

    private EditText textQuery;
    private TextView result;
    private TextView pageIdx;
    private Button btnPrev;
    private Button btnNext;

    private final ExecutorService executor = Executors.newSingleThreadExecutor();

    private String[] texts = new String[0];
    private int idx = 0;
    private Exception error = null;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {

        View view = inflater.inflate(R.layout.fragment_query, container, false);
        textQuery = view.findViewById(R.id.textQuery);
        result = view.findViewById(R.id.result);
        pageIdx = view.findViewById(R.id.pageidx);
        btnPrev = view.findViewById(R.id.btnPrev);
        btnNext = view.findViewById(R.id.btnNext);
        Button btnSearch = view.findViewById(R.id.btnSearch);

        // Assign listeners
        btnSearch.setOnClickListener(v -> search());
        btnPrev.setOnClickListener(v -> changeText(-1));
        btnNext.setOnClickListener(v -> changeText(1));

        return view;
    }

    /**
     * Perform the search.
     */
    private void search() {

        error = null;
        result.setText("");

        String queryText = textQuery.getText().toString().trim();
        if (queryText.isEmpty()) return;

        executor.execute(() -> {
            try {
                texts = RAG.queryRAG(queryText);
            } catch (Exception e) {
                error = e;
                texts = new String[0];
            }

            requireActivity().runOnUiThread(() -> {
                idx = 0;
                updateUI();
            });
        });
    }

    /**
     * Navigate to the given text.
     * @param offset The offset to apply to the current index (e.g., -1 for previous, +1 for next).
     */
    private void changeText(int offset) {
        idx = Math.max(0, Math.min(idx + offset, texts.length - 1));
        updateUI();
    }

    private void updateUI() {

        // On empty results, display a message
        if (texts.length == 0) {
            result.setText(
                    error == null ?
                    "No results found in the vector index." :
                    "Error during search: " + error.getMessage());
            pageIdx.setText("0 / 0");
            btnPrev.setEnabled(false);
            btnNext.setEnabled(false);
            return;
        }

        // Else, display the chunk
        result.setText(texts[idx]);
        pageIdx.setText((idx + 1) + " / " + texts.length);
        btnPrev.setEnabled(idx > 0);
        btnNext.setEnabled(idx < texts.length - 1);
    }
}