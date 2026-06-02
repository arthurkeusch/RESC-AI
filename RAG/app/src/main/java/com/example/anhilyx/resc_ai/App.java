package com.example.anhilyx.resc_ai;

import android.app.Application;
import io.objectbox.BoxStore;

public class App extends Application {
    private static BoxStore boxStore;

    @Override
    public void onCreate() {
        super.onCreate();
        // MyObjectBox est généré automatiquement après le "Make Project"
        boxStore = MyObjectBox.builder()
                .androidContext(this)
                .build();
    }

    public static BoxStore getBoxStore() {
        return boxStore;
    }
}
