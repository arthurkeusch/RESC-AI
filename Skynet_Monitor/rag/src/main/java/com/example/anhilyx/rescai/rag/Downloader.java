package com.example.anhilyx.rescai.rag;

import android.util.Log;

import androidx.annotation.NonNull;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;

/**
 * Utility class for downloading files from the internet.
 * This class can be re-used for more specific use cases.
 */
public class Downloader {

    protected static final int BUFFER_SIZE = 8192;
    protected static final OkHttpClient CLIENT = new OkHttpClient();

    /**
     * Callback for the download process.
     */
    public interface DownloadCallback {

        /**
         * Called when the download is successful.
         */
        void onSuccess();

        /**
         * Called when the download fails.
         * @param code The error code.
         * @param message The error message.
         */
        void onError(int code, String message);

        /**
         * Called to report the download progress.
         * @param progress The download progress as a float between `0` and `1`.
         */
        void onProgress(float progress);
    }

    /**
     * Downloads a file from the given URL and saves it to the specified output file.
     * @param url The URL to download the file from.
     * @param outFile The file to save the downloaded content to.
     * @param callback The callbacks to be invoked during the download.
     */
    public static void downloadFile(String url, File outFile, DownloadCallback callback) {
        CLIENT
                .newCall(
                        new Request.Builder().url(url).build()
                )
                .enqueue(
                        new Callback() {
                            @Override
                            public void onFailure(@NonNull Call call, @NonNull IOException e) {

                                // On failure, we return an internal error with the exception message
                                callback.onError(500, e.getMessage());
                            }

                            @Override
                            public void onResponse(@NonNull Call call, @NonNull Response response) {

                                // On unsuccessful requests, return the error code and message from the response
                                if (!response.isSuccessful()) {
                                    callback.onError(response.code(), response.message());
                                    return;
                                }

                                // On successful response, progressively write the response body to the output file, while reporting progress
                                try (
                                        InputStream is = response.body().byteStream();
                                        FileOutputStream fos = new FileOutputStream(outFile)
                                ) {

                                    byte[] buffer = new byte[BUFFER_SIZE];
                                    int read;
                                    long contentLength = Math.max(response.body().contentLength(), 1);  // Avoid division by zero, and avoid checking every loop iteration if contentLength is valid
                                    long bytesRead = 0;
                                    callback.onProgress(0.0f);  // Start with 0% progress

                                    while ((read = is.read(buffer)) != -1) {
                                        fos.write(buffer, 0, read);
                                        bytesRead += read;
                                        callback.onProgress((float) bytesRead / contentLength);
                                    }

                                    callback.onProgress(1.0f);  // Ensure we report 100% progress at the end
                                    callback.onSuccess();
                                }

                                // If an exception occurs during file writing, we return an internal error with the exception message
                                catch (IOException e) {
                                    callback.onError(500, e.getMessage());
                                }
                            }
                        }
                );
    }
}
