package com.example.anhilyx.rescai.rag;

import android.util.Pair;

import java.io.File;
import java.util.ArrayList;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Utility class for downloading models from Hugging Face.
 */
public class HFDownloader {

    /**
     * Downloads the model (and additional required files) from Hugging Face for the given repository ID and saves them to the specified directory.
     * @param directory The directory to save the downloaded files to.
     * @param repoId The repository ID on Hugging Face.
     * @param callback The callbacks to be invoked during the download.
     */
    public static void downloadRepository(File directory, String repoId, Downloader.DownloadCallback callback) {

        // Construct the list of files to download (model and tokenizer) along with their output file paths
        ArrayList<Pair<String, File>> args = new ArrayList<>();
        args.add(new Pair<>(
                "https://huggingface.co/" + repoId + "/resolve/main/onnx/model.onnx?download=true",
                new File(directory, "model.onnx")
        ));
        args.add(new Pair<>(
                "https://huggingface.co/" + repoId + "/resolve/main/tokenizer.json?download=true",
                new File(directory, "tokenizer.json")
        ));

        // Create a custom callback to handle the progress of multiple files
        final AtomicBoolean error = new AtomicBoolean(false);
        final AtomicInteger filesProgress = new AtomicInteger(0);
        final AtomicBoolean success = new AtomicBoolean(false);
        Downloader.DownloadCallback _callback = new Downloader.DownloadCallback() {
            @Override
            public void onSuccess() {
                success.set(true);
            }

            @Override
            public void onError(int code, String message) {
                callback.onError(code, message);
                error.set(true);
            }

            @Override
            public void onProgress(float progress) {
                callback.onProgress(progress / args.size() + (filesProgress.get() / (float) args.size()));
            }
        };

        // Download all files
        for (Pair<String, File> arg : args) {
            success.set(false);
            Downloader.downloadFile(arg.first, arg.second, _callback);
            while (!success.get() && !error.get()) {
                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    callback.onError(500, "Download interrupted: " + e.getMessage());
                    error.set(true);
                    break;
                }
            }
            filesProgress.incrementAndGet();
            if (error.get()) {
                return;
            }
        }
        callback.onSuccess();
    }
}
