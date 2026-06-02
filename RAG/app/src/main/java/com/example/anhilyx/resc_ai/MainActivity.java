package com.example.anhilyx.resc_ai;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Bundle;

import androidx.appcompat.app.AppCompatActivity;

import com.google.android.material.tabs.TabLayout;

import java.util.List;

public class MainActivity extends AppCompatActivity {

    private TabsWrapper layout;

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        layout = findViewById(R.id.tabs);

        // Define tabs
        layout.setup(this, List.of(new TabConfig[]{
                new TabConfig("RAG Creation", new RagCreationFragment()),
                new TabConfig("RAG Extraction", new RagExtractionFragment())
        }));

        // Retrieve variables
        SharedPreferences prefs = getSharedPreferences("rag_prefs", Context.MODE_PRIVATE);
        boolean isRagCreated = prefs.getBoolean("is_rag_created", false);

        // Select the right initial tab
        if (!isRagCreated) {
            layout.getViewPager().setCurrentItem(0, false);
            setExtractionTabEnabled(false);
        } else {
            layout.getViewPager().setCurrentItem(1, false);
            setExtractionTabEnabled(true);
        }
    }

    /**
     * Enable/disable the extraction tab.
     * @param enabled True to enable, false to disable.
     */
    private void setExtractionTabEnabled(boolean enabled) {
        layout.getViewPager().setUserInputEnabled(enabled);
        layout.getTabs().post(() -> {
            TabLayout.Tab extractionTab = layout.getTabs().getTabAt(1);
            if (extractionTab != null) {
                extractionTab.view.setEnabled(enabled);
                extractionTab.view.setAlpha(enabled ? 1.0f : 0.4f);
            }
        });
    }

    /**
     * Callback when the RAG is created.
     */
    public void onRagCreated() {
        setExtractionTabEnabled(true);
        layout.getViewPager().setCurrentItem(1, true);
    }
}