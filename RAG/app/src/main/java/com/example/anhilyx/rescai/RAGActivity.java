package com.example.anhilyx.rescai;

import android.os.Bundle;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.fragment.app.Fragment;
import androidx.viewpager2.adapter.FragmentStateAdapter;
import androidx.viewpager2.widget.ViewPager2;

import com.example.anhilyx.rescai.rag.RAG;
import com.google.android.material.tabs.TabLayout;
import com.google.android.material.tabs.TabLayoutMediator;

public class RAGActivity extends AppCompatActivity {

    protected TabLayout tabs;
    protected ViewPager2 view;

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_rag);
        tabs = findViewById(R.id.tabs);
        view = findViewById(R.id.view);

        // Define which fragments to show in each tab
        view.setAdapter(new FragmentStateAdapter(this) {
            @NonNull
            @Override
            public Fragment createFragment(int position) {
                switch (position) {
                    case 0:
                        return new BuildFragment();
                    case 1:
                        return new QueryFragment();
                    default:
                        throw new IndexOutOfBoundsException("Invalid tab position: " + position);
                }
            }

            @Override
            public int getItemCount() {
                return 2;
            }
        });

        // Apply titles to the tabs
        new TabLayoutMediator(this.tabs, view, (tab, position) -> {
            switch (position) {
                case 0:
                    tab.setText("Build");
                    break;
                case 1:
                    tab.setText("Query");
                    break;
                default:
                    throw new IndexOutOfBoundsException("Invalid tab position: " + position);
            }
        }).attach();

        // Check if the RAG index is already built
        boolean isCreated = false;
        try {
            isCreated = RAG.queryRAG("").length > 0;
        } catch (Exception ignored) {}

        // If the RAG index isn't built, start on the Build tab, otherwise start on the Query tab
        view.setCurrentItem(isCreated ? 1 : 0, false);

    }

    /**
     * Enable or disable the Query tab and its content.
     * @param enabled True to enable the Query tab, false to disable it.
     */
    protected void toggleQueryTab(boolean enabled) {
        view.setUserInputEnabled(enabled);
        tabs.post(() -> {
            TabLayout.Tab queryTab = tabs.getTabAt(1);
            if (queryTab != null) {
                queryTab.view.setEnabled(enabled);
                queryTab.view.setAlpha(enabled ? 1.0f : 0.5f);
            }
        });
    }
}