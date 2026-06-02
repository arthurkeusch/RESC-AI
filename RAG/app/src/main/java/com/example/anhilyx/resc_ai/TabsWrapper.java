package com.example.anhilyx.resc_ai;

import android.content.Context;
import android.util.AttributeSet;
import android.widget.LinearLayout;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentActivity;
import androidx.viewpager2.adapter.FragmentStateAdapter;
import androidx.viewpager2.widget.ViewPager2;
import com.google.android.material.tabs.TabLayout;
import com.google.android.material.tabs.TabLayoutMediator;
import java.util.List;

public class TabsWrapper extends LinearLayout {

    private TabLayout tabs;
    private ViewPager2 pager;

    public TabsWrapper(Context context) {
        super(context);
        init(context);
    }

    public TabsWrapper(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        init(context);
    }

    public TabsWrapper(Context context, @Nullable AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        init(context);
    }

    private void init(Context context) {
        setOrientation(VERTICAL);
        inflate(context, R.layout.wrapper_tabs, this);
        tabs = findViewById(R.id.tab_layout);
        pager = findViewById(R.id.pager);
    }

    /**
     * Setup the tabs and the view pager.
     * @param hostActivity The activity hosting the tabs.
     * @param tabs A list of tab configurations.
     */
    public void setup(FragmentActivity hostActivity, List<TabConfig> tabs) {

        // Create the adapter from the list of tabs
        pager.setAdapter(new FragmentStateAdapter(hostActivity) {
            @NonNull
            @Override
            public Fragment createFragment(int position) {
                return tabs.get(position).getFragment();
            }

            @Override
            public int getItemCount() {
                return tabs.size();
            }
        });

        // Apply titles to the tabs
        new TabLayoutMediator(this.tabs, pager, (tab, position) -> {
            tab.setText(tabs.get(position).getTitle());
        }).attach();
    }

    // Expose elements to the outside world

    /**
     * Get the view pager.
     * @return The view pager.
     */
    public ViewPager2 getViewPager() { return pager; }

    /**
     * Get the tabs.
     * @return The tabs.
     */
    public TabLayout getTabs() { return tabs; }
}