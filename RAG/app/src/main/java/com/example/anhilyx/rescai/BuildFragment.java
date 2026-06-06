package com.example.anhilyx.rescai;

import android.net.Uri;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.TextView;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;

import com.example.anhilyx.rescai.rag.RAG;
import com.google.android.material.progressindicator.LinearProgressIndicator;

import java.io.FileNotFoundException;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class BuildFragment extends Fragment {

    private Button selector;
    private LinearProgressIndicator progress;
    private TextView status;

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final ActivityResultLauncher<String> pickPDFLauncher = registerForActivityResult(
            new ActivityResultContracts.GetContent(), this::processPDF
    );

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {

        View view = inflater.inflate(R.layout.fragment_build, container, false);
        selector = view.findViewById(R.id.selector);
        progress = view.findViewById(R.id.progress);
        status = view.findViewById(R.id.status);

        selector.setOnClickListener(v -> pickPDFLauncher.launch("application/pdf"));

        return view;
    }

    /**
     * Process the selected PDF file.
     * @param uri The URI of the selected PDF file.
     */
    private void processPDF(Uri uri) {

        // If the user canceled the file selection, do nothing
        if (uri == null) return;

        progress.setVisibility(View.VISIBLE);
        selector.setEnabled(false);

        executor.execute(() -> {
                try {
                    RAG.createRAG(requireActivity().getContentResolver().openInputStream(uri), new RAG.RAGCallback() {
                        @Override
                        public void onSuccess() {
                            status.setText("Done.");
                            RAGActivity activity = (RAGActivity) requireActivity();
                            activity.toggleQueryTab(true);
                            activity.view.setCurrentItem(1, true);
                            selector.setEnabled(true);
                        }

                        @Override
                        public void onError(Exception e, int step) {
                            status.setText("Error building RAG: " + e.getMessage());
                            selector.setEnabled(true);
                        }

                        @Override
                        public void onProgress(float progressValue, int step) {
                            requireActivity().runOnUiThread(() -> {
                                if (step == RAG.RAGCallback.STEP_READING_PDF) {
                                    status.setText("Reading PDF...");
                                } else if (step == RAG.RAGCallback.STEP_CHUNKING_PDF) {
                                    status.setText("Chunking PDF...");
                                } else if (step == RAG.RAGCallback.STEP_CREATING_RAG) {
                                    status.setText(String.format(Locale.US, "Creating RAG index... %.1f%%", 100.0f * progressValue));
                                }
                            });
                        }
                    });

                } catch (FileNotFoundException e) {
                    status.setText("Error opening PDF: " + e.getMessage());
                    selector.setEnabled(true);
                }
        });
    }
}