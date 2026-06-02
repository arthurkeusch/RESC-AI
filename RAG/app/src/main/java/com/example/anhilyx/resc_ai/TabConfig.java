package com.example.anhilyx.resc_ai;

import androidx.fragment.app.Fragment;

public class TabConfig {

    private final String title;
    private final Fragment fragment;

    /**
     * TabConfig constructor.
     * @param title The title of the tab.
     * @param fragment The fragment associated with the tab.
     */
    public TabConfig(String title, Fragment fragment) {
        this.title = title;
        this.fragment = fragment;
    }

    /**
     * Get the title of the tab.
     * @return The title of the tab.
     */
    public String getTitle() {
        return title;
    }

    /**
     * Get the fragment associated with the tab.
     * @return The fragment associated with the tab.
     */
    public Fragment getFragment() {
        return fragment;
    }
}
