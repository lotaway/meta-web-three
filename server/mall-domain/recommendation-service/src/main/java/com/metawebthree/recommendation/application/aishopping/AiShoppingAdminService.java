package com.metawebthree.recommendation.application.aishopping;

import com.metawebthree.recommendation.domain.aishopping.entity.AiSearchLog;
import com.metawebthree.recommendation.domain.aishopping.entity.AiShoppingConfig;
import com.metawebthree.recommendation.domain.aishopping.entity.IndexStatus;
import com.metawebthree.recommendation.domain.aishopping.repository.AiSearchLogRepository;
import com.metawebthree.recommendation.domain.aishopping.repository.AiShoppingConfigRepository;
import com.metawebthree.recommendation.domain.aishopping.service.AiShoppingIndexService;
import com.metawebthree.recommendation.infrastructure.aishopping.provider.AiProviderClient;
import java.util.List;
import java.util.Map;
import org.springframework.stereotype.Service;

/** Backend AI shopping admin service: config CRUD, index rebuild/status, provider connectivity, search logs. */
@Service
public class AiShoppingAdminService {

    private final AiShoppingConfigRepository configRepository;
    private final AiSearchLogRepository logRepository;
    private final AiShoppingIndexService indexService;
    private final AiProviderConfig providerConfig;
    private final AiProviderClient providerClient;
    private final AiShoppingFeatureGuard featureGuard;

    public AiShoppingAdminService(AiShoppingConfigRepository configRepository,
                                  AiSearchLogRepository logRepository,
                                  AiShoppingIndexService indexService,
                                  AiProviderConfig providerConfig,
                                  AiProviderClient providerClient,
                                  AiShoppingFeatureGuard featureGuard) {
        this.configRepository = configRepository;
        this.logRepository = logRepository;
        this.indexService = indexService;
        this.providerConfig = providerConfig;
        this.providerClient = providerClient;
        this.featureGuard = featureGuard;
    }

    // ==================== Config ====================

    public List<AiShoppingConfig> listConfigs() {
        return configRepository.findAll();
    }

    public AiShoppingConfig saveConfig(String key, String value, String description) {
        if (key == null || key.isBlank()) {
            throw new IllegalArgumentException("config key is required");
        }
        return configRepository.save(new AiShoppingConfig(key, value, description));
    }

    public void deleteConfig(String key) {
        configRepository.delete(key);
    }

    // ==================== Index ====================

    public void rebuildIndex(String type) {
        featureGuard.requireEnabled();
        indexService.rebuild(type);
    }

    public Map<String, Object> indexStatus() {
        AiProviderSettings settings = providerConfig.getSettings();
        return Map.of(
                "status", indexService.getStatus().toMap(),
                "vectorStore", settings.getVectorStore(),
                "collectionText", settings.getMilvusCollectionText(),
                "collectionImage", settings.getMilvusCollectionImage());
    }

    public boolean indexRunning() {
        return indexService.isRunning();
    }

    // ==================== Provider test ====================

    public Map<String, Object> testProvider(String type) {
        featureGuard.requireEnabled();
        String normalized = type == null ? "embedding" : type.toLowerCase();
        try {
            long start = System.currentTimeMillis();
            switch (normalized) {
                case "image":
                    providerClient.embedImage(new byte[]{(byte) 0xFF, (byte) 0xD8, (byte) 0xFF, (byte) 0xE0});
                    break;
                case "llm":
                    providerClient.chat("You are a test assistant", "Reply OK");
                    break;
                default:
                    providerClient.embedText("test");
                    break;
            }
            return Map.of("success", true, "type", normalized,
                    "responseTimeMs", System.currentTimeMillis() - start);
        } catch (Exception e) {
            return Map.of("success", false, "type", normalized, "error", e.getMessage());
        }
    }

    // ==================== Logs ====================

    public List<AiSearchLog> recentLogs(int limit) {
        return logRepository.findRecent(limit <= 0 ? 50 : limit);
    }
}
