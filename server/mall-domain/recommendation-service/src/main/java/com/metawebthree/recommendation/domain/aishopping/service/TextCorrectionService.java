package com.metawebthree.recommendation.domain.aishopping.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.recommendation.domain.aishopping.entity.TextCorrection;
import com.metawebthree.recommendation.infrastructure.aishopping.provider.AiProviderClient;
import com.metawebthree.recommendation.infrastructure.aishopping.product.ProductCatalogCache;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

/**
 * Text correction: prefers the LLM, falls back to local dictionary correction
 * when the LLM is unavailable or fails.
 */
@Service
public class TextCorrectionService {

    private static final Logger log = LoggerFactory.getLogger(TextCorrectionService.class);

    private static final String SYSTEM_PROMPT =
            "You are a Chinese e-commerce shopping search correction assistant. "
            + "User input may contain typos, pinyin, homophones, simplified/traditional "
            + "mixing, or extra/missing characters. Output the corrected shopping keyword. "
            + "Return JSON only without extra explanation, in the format "
            + "{\"corrected\":\"corrected keyword\",\"suggestions\":[\"suggestion 1\",\"suggestion 2\"]}. "
            + "If no correction is needed, corrected equals the original keyword and "
            + "suggestions is an empty array.";

    private final AiProviderClient providerClient;
    private final LocalTextCorrector localCorrector;
    private final ObjectMapper objectMapper = new ObjectMapper();

    public TextCorrectionService(AiProviderClient providerClient, ProductCatalogCache catalogCache) {
        this.providerClient = providerClient;
        this.localCorrector = new LocalTextCorrector(catalogCache);
    }

    public TextCorrection correct(String query) {
        if (query == null || query.trim().isEmpty()) {
            TextCorrection none = new TextCorrection();
            none.setOriginal(query == null ? "" : query);
            none.setCorrected(query == null ? "" : query);
            none.setChanged(false);
            none.setSource(TextCorrection.CorrectionSource.NONE);
            return none;
        }

        TextCorrection llmResult = tryLlm(query);
        if (llmResult != null) {
            return llmResult;
        }

        return localCorrect(query);
    }

    private TextCorrection tryLlm(String query) {
        try {
            String content = providerClient.chat(SYSTEM_PROMPT, "Search query: " + query);
            return parseLlmResult(query, content);
        } catch (Exception e) {
            log.warn("LLM text correction failed, fallback to local: {}", e.getMessage());
            return null;
        }
    }

    private TextCorrection parseLlmResult(String query, String content) throws Exception {
        JsonNode node = objectMapper.readTree(extractJson(content));
        String corrected = node.path("corrected").asText();
        List<String> suggestions = new ArrayList<>();
        node.path("suggestions").forEach(s -> suggestions.add(s.asText()));

        if (corrected == null || corrected.isBlank()) {
            return null;
        }

        TextCorrection result = new TextCorrection();
        result.setOriginal(query);
        result.setCorrected(corrected);
        result.setChanged(!query.equals(corrected));
        result.setSuggestions(suggestions);
        result.setSource(TextCorrection.CorrectionSource.LLM);
        return result;
    }

    private TextCorrection localCorrect(String query) {
        String best = localCorrector.bestCorrection(query);
        List<String> suggestions = localCorrector.suggest(query);

        TextCorrection result = new TextCorrection();
        result.setOriginal(query);
        result.setCorrected(best != null ? best : query);
        result.setChanged(best != null && !query.equals(best));
        result.setSuggestions(suggestions);
        result.setSource(best != null
                ? TextCorrection.CorrectionSource.LOCAL
                : TextCorrection.CorrectionSource.NONE);
        return result;
    }

    private String extractJson(String content) {
        String trimmed = content.trim();
        if (trimmed.startsWith("```")) {
            int firstNewline = trimmed.indexOf('\n');
            int lastBacktick = trimmed.lastIndexOf("```");
            if (firstNewline >= 0 && lastBacktick > firstNewline) {
                trimmed = trimmed.substring(firstNewline + 1, lastBacktick).trim();
            }
        }
        int start = trimmed.indexOf('{');
        int end = trimmed.lastIndexOf('}');
        if (start >= 0 && end > start) {
            return trimmed.substring(start, end + 1);
        }
        return trimmed;
    }
}
