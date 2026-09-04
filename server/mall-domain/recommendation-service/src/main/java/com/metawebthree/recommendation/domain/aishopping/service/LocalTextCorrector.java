package com.metawebthree.recommendation.domain.aishopping.service;

import com.metawebthree.recommendation.infrastructure.aishopping.product.ProductCatalogCache;
import com.metawebthree.recommendation.infrastructure.aishopping.product.ProductDataProvider;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Local text-correction fallback based on product keyword dictionary: character
 * bigram similarity plus edit-distance scoring. Used when the LLM is unavailable.
 */
public class LocalTextCorrector {

    private static final double MIN_SIMILARITY = 0.55;

    private final ProductCatalogCache catalogCache;

    public LocalTextCorrector(ProductCatalogCache catalogCache) {
        this.catalogCache = catalogCache;
    }

    /**
     * Returns candidate corrections. An exact dictionary hit returns only the
     * query itself; an empty result means no suggestion.
     */
    public List<String> suggest(String query) {
        String normalized = normalize(query);
        if (normalized.isBlank()) {
            return List.of();
        }
        List<String> dictionary = buildDictionary();
        if (dictionary.isEmpty()) {
            return List.of();
        }
        Set<String> normalizedDictionary = new LinkedHashSet<>();
        for (String word : dictionary) {
            normalizedDictionary.add(normalize(word));
        }
        if (normalizedDictionary.contains(normalized)) {
            return List.of(normalized);
        }
        return rankCandidates(normalized, dictionary);
    }

    private List<String> rankCandidates(String normalized, List<String> dictionary) {
        Set<String> result = new LinkedHashSet<>();
        for (String candidate : dictionary) {
            String normCandidate = normalize(candidate);
            if (normCandidate.isBlank() || normCandidate.equals(normalized)) {
                continue;
            }
            if (similarity(normalized, normCandidate) >= MIN_SIMILARITY) {
                result.add(normCandidate);
            }
        }
        return result.stream()
                .sorted(Comparator.comparingDouble((String s) -> -similarity(normalized, s)))
                .limit(5)
                .collect(Collectors.toList());
    }

    /**
     * Returns the original word on an exact dictionary hit, otherwise the most
     * similar candidate (or null when below the threshold).
     */
    public String bestCorrection(String query) {
        String normalized = normalize(query);
        if (normalized.isBlank()) {
            return null;
        }
        List<String> suggestions = suggest(normalized);
        if (suggestions.isEmpty()) {
            return null;
        }
        return suggestions.get(0);
    }

    private double similarity(String a, String b) {
        return Math.max(charBigramSimilarity(a, b), tokenOverlapSimilarity(a, b));
    }

    private List<String> buildDictionary() {
        Set<String> words = new LinkedHashSet<>();
        for (ProductDataProvider.ProductItem item : catalogCache.all()) {
            if (item.name != null) {
                words.add(item.name.trim());
            }
            if (item.subTitle != null) {
                words.add(item.subTitle.trim());
            }
            if (item.sku != null && !item.sku.isBlank()) {
                words.add(item.sku.trim());
            }
        }
        return new ArrayList<>(words);
    }

    private double charBigramSimilarity(String a, String b) {
        if (a.isEmpty() && b.isEmpty()) {
            return 1.0;
        }
        if (a.isEmpty() || b.isEmpty()) {
            return 0.0;
        }
        Set<String> bigramsA = bigrams(a);
        Set<String> bigramsB = bigrams(b);
        if (bigramsA.isEmpty() || bigramsB.isEmpty()) {
            return a.equals(b) ? 1.0 : 0.0;
        }
        long intersection = bigramsA.stream().filter(bigramsB::contains).count();
        return 2.0 * intersection / (bigramsA.size() + bigramsB.size());
    }

    private Set<String> bigrams(String text) {
        Set<String> result = new LinkedHashSet<>();
        for (int i = 0; i + 1 < text.length(); i++) {
            result.add(text.substring(i, i + 2));
        }
        if (text.length() == 1) {
            result.add(text);
        }
        return result;
    }

    private double tokenOverlapSimilarity(String a, String b) {
        String[] tokensA = a.split("\\s+");
        String[] tokensB = b.split("\\s+");
        if (tokensA.length == 0 || tokensB.length == 0) {
            return 0.0;
        }
        Set<String> setA = Set.of(tokensA);
        long hits = 0;
        for (String token : tokensB) {
            if (setA.contains(token)) {
                hits++;
            }
        }
        return 2.0 * hits / (tokensA.length + tokensB.length);
    }

    private String normalize(String text) {
        if (text == null) {
            return "";
        }
        return fullWidthToHalfWidth(text).trim().toLowerCase();
    }

    private String fullWidthToHalfWidth(String text) {
        StringBuilder sb = new StringBuilder();
        for (char c : text.toCharArray()) {
            if (c >= '\uFF01' && c <= '\uFF5E') {
                sb.append((char) (c - 0xFEE0));
            } else if (c == '\u3000') {
                sb.append(' ');
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }
}
