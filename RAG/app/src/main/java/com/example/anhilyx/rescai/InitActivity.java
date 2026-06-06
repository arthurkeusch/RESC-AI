package com.example.anhilyx.rescai;

import android.content.Intent;
import android.os.Bundle;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import com.example.anhilyx.rescai.rag.RAG;
import com.google.android.material.progressindicator.LinearProgressIndicator;

import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class InitActivity extends AppCompatActivity {

    protected final ExecutorService executor = Executors.newSingleThreadExecutor();

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_init);
        TextView statusTextView = findViewById(R.id.status);
        LinearProgressIndicator progressBar = findViewById(R.id.progress);

        executor.execute(() ->
                RAG.init(this, new RAG.RAGCallback() {

                    // On success, move to the RAGActivity
                    @Override
                    public void onSuccess() {
                        Intent intent = new Intent(InitActivity.this, RAGActivity.class);
                        startActivity(intent);
                        finish();
                    }

                    // On error, display the error in ErrorActivity
                    @Override
                    public void onError(Exception e, int step) {
                        Intent intent = new Intent(InitActivity.this, ErrorActivity.class);
                        intent.putExtra("error_message", e.toString());
                        startActivity(intent);
                        finish();
                    }

                    @Override
                    public void onProgress(float progress, int step) {
                        if (step == RAG.RAGCallback.STEP_INITIALIZING) {
                            statusTextView.setText("Status: Initializing...");
                        } else if (step == RAG.RAGCallback.STEP_INSTALLING_MODEL) {
                            statusTextView.setText(String.format(Locale.US, "Status: Downloading model... %.1f%%", 100.0f * progress));

                            progressBar.setProgress((int) (1000.0f * progress), true);
                        }
                    }
                }, false)
        );
    }
}