package com.example.anhilyx.rescai.rag;

import com.tom_roush.pdfbox.pdmodel.PDDocument;
import com.tom_roush.pdfbox.text.PDFTextStripper;

import java.io.IOException;
import java.io.InputStream;
import java.text.BreakIterator;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Utility class for extracting text chunks from a PDF document.
 */
public class PDFExtractor {

    protected final String content;

    /**
     * Constructor for the Extractor class.
     * @param pdfIS The input stream of the PDF file to be processed.
     * @throws IOException If there is an error reading the PDF file or extracting text from it.
     */
    public PDFExtractor(InputStream pdfIS) throws IOException {

        PDDocument document = PDDocument.load(pdfIS);
        PDFTextStripper stripper = new PDFTextStripper();
        content = stripper.getText(document);
        document.close();
    }

    /**
     * Extract text chunks from the PDF document content.
     * @return An array of text chunks extracted from the PDF document, where each chunk is a string containing one or more sentences.
     */
    public String[] extractChunks() {

        // Clean the content by removing hyphenation and normalizing whitespace
        String text = content
                .replaceAll("-\\s*\\r?\\n\\s*", "")
                .replaceAll("[\\r\\n\\t]+", " ")
                .replaceAll("\\s+", " ");

        // Extract all sentences
        List<String> sentences = new ArrayList<>();
        BreakIterator boundary = BreakIterator.getSentenceInstance(Locale.getDefault());
        boundary.setText(text);
        int start = boundary.first();
        for (int end = boundary.next(); end != BreakIterator.DONE; start = end, end = boundary.next()) {
            String sentence = text.substring(start, end);
            if (!sentence.trim().isEmpty()) {
                sentences.add(sentence);
            }
        }

        // Build overlapping chunks
        ArrayList<String> chunks = new ArrayList<>();
        int idx = 0;
        int n = sentences.size();
        while (idx < n) {
            StringBuilder chunk = new StringBuilder();
            int chunkSize = 0;
            int nSentences = 0;

            while (idx < n && chunkSize < Config.CHUNK_SIZE) {
                String sentence = sentences.get(idx);
                chunkSize += sentence.length();
                chunk.append(sentence);
                idx++;
                nSentences++;
            }

            chunks.add(chunk.toString());
            if (nSentences > 1) { idx--; }  // Include the previous sentence in the next chunk, unless the sentence is the whole chunk
        }

        return chunks.toArray(new String[0]);
    }
}
