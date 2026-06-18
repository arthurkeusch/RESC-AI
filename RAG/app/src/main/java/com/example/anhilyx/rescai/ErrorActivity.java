package com.example.anhilyx.rescai;

import android.os.Bundle;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

public class ErrorActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_error);
        TextView errorTextView = findViewById(R.id.error);
        String errorMessage = getIntent().getStringExtra("error_message");
        errorTextView.setText(errorMessage);
    }
}