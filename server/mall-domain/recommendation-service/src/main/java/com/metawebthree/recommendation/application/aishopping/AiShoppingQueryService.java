package com.metawebthree.recommendation.application.aishopping;

import com.metawebthree.recommendation.domain.aishopping.entity.AiProductMatch;
import com.metawebthree.recommendation.domain.aishopping.entity.AiSearchLog;
import com.metawebthree.recommendation.domain.aishopping.entity.TextCorrection;
import com.metawebthree.recommendation.domain.aishopping.repository.AiSearchLogRepository;
import com.metawebthree.recommendation.domain.aishopping.service.ImageSearchService;
import com.metawebthree.recommendation.domain.aishopping.service.SmartMatchService;
import com.metawebthree.recommendation.domain.aishopping.service.TextCorrectionService;
import java.util.List;
import java.util.Map;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

/** Customer-side AI shopping query service. */
@Slf4j
@Service
public class AiShoppingQueryService {

    private final TextCorrectionService textCorrectionService;
    private final SmartMatchService smartMatchService;
    private final ImageSearchService imageSearchService;
    private final AiSearchLogRepository logRepository;
    private final AiProviderConfig providerConfig;

    public AiShoppingQueryService(TextCorrectionService textCorrectionService,
                                  SmartMatchService smartMatchService,
                                  ImageSearchService imageSearchService,
                                  AiSearchLogRepository logRepository,
                                  AiProviderConfig providerConfig) {
        this.textCorrectionService = textCorrectionService;
        this.smartMatchService = smartMatchService;
        this.imageSearchService = imageSearchService;
        this.logRepository = logRepository;
        this.providerConfig = providerConfig;
    }

    public TextCorrection correctText(String text) {
        return textCorrectionService.correct(text);
    }

    public List<AiProductMatch> smartMatch(String query, Integer topK) {
        int k = topK != null && topK > 0 ? topK : providerConfig.getSettings().getDefaultTopK();
        long start = System.currentTimeMillis();
        List<AiProductMatch> matches = smartMatchService.match(query, k);
        saveLog(AiSearchLog.SearchType.SMART_MATCH, null, query, null, matches.size(),
                System.currentTimeMillis() - start);
        return matches;
    }

    public List<AiProductMatch> imageSearch(byte[] imageBytes, Integer topK) {
        int k = topK != null && topK > 0 ? topK : providerConfig.getSettings().getDefaultTopK();
        long start = System.currentTimeMillis();
        List<AiProductMatch> matches = imageSearchService.search(imageBytes, k);
        saveLog(AiSearchLog.SearchType.IMAGE_SEARCH, null, "[image]", null, matches.size(),
                System.currentTimeMillis() - start);
        return matches;
    }

    /** One-stop search: correction then smart match. Returns correction plus matched products. */
    public Map<String, Object> combinedSearch(String query, Integer topK, Long userId) {
        int k = topK != null && topK > 0 ? topK : providerConfig.getSettings().getDefaultTopK();
        long start = System.currentTimeMillis();

        TextCorrection correction = textCorrectionService.correct(query);
        String effectiveQuery = correction.isChanged() && correction.getCorrected() != null
                ? correction.getCorrected() : query;
        List<AiProductMatch> matches = smartMatchService.match(effectiveQuery, k);

        saveLog(AiSearchLog.SearchType.COMBINED_SEARCH, userId, query, correction.getCorrected(),
                matches.size(), System.currentTimeMillis() - start);

        return Map.of(
                "query", query,
                "correction", correction,
                "matches", matches);
    }

    private void saveLog(AiSearchLog.SearchType type, Long userId, String query, String corrected,
                         int resultCount, long responseTimeMs) {
        try {
            AiSearchLog log = new AiSearchLog();
            log.setUserId(userId);
            log.setSearchType(type);
            log.setQueryText(query);
            log.setCorrectedText(corrected);
            log.setResultCount(resultCount);
            log.setResponseTimeMs(responseTimeMs);
            logRepository.save(log);
        } catch (Exception e) {
            log.warn("Failed to save AI shopping search log", e);
        }
    }
}
