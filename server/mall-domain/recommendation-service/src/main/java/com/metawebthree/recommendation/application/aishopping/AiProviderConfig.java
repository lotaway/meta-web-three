package com.metawebthree.recommendation.application.aishopping;

import com.metawebthree.recommendation.domain.aishopping.repository.AiShoppingConfigRepository;
import com.metawebthree.recommendation.infrastructure.aishopping.config.AiShoppingProperties;
import java.util.function.Consumer;
import org.springframework.stereotype.Service;

/**
 * Resolves the effective AI provider config: application.yml defaults plus
 * ai_shopping_config DB overrides.
 */
@Service
public class AiProviderConfig {

    public static final String KEY_EMBEDDING_BASE_URL = "embedding.base-url";
    public static final String KEY_EMBEDDING_API_KEY = "embedding.api-key";
    public static final String KEY_EMBEDDING_MODEL = "embedding.model";
    public static final String KEY_IMAGE_BASE_URL = "image-embedding.base-url";
    public static final String KEY_IMAGE_API_KEY = "image-embedding.api-key";
    public static final String KEY_IMAGE_MODEL = "image-embedding.model";
    public static final String KEY_LLM_BASE_URL = "llm.base-url";
    public static final String KEY_LLM_API_KEY = "llm.api-key";
    public static final String KEY_LLM_MODEL = "llm.model";
    public static final String KEY_VECTOR_STORE = "vector.store";
    public static final String KEY_MILVUS_HOST = "milvus.host";
    public static final String KEY_MILVUS_PORT = "milvus.port";
    public static final String KEY_MILVUS_COLLECTION_TEXT = "milvus.collection-text";
    public static final String KEY_MILVUS_COLLECTION_IMAGE = "milvus.collection-image";
    public static final String KEY_INDEX_STATUS = "index.status";
    public static final String KEY_INDEX_LAST_REBUILT = "index.last-rebuilt-at";
    public static final String KEY_ENABLED = "ai-shopping.enabled";

    private final AiShoppingProperties properties;
    private final AiShoppingConfigRepository configRepository;

    public AiProviderConfig(AiShoppingProperties properties, AiShoppingConfigRepository configRepository) {
        this.properties = properties;
        this.configRepository = configRepository;
    }

    public boolean isEnabled() {
        return configRepository.findValue(KEY_ENABLED)
                .map(Boolean::parseBoolean)
                .orElse(properties.isEnabled());
    }

    public AiProviderSettings getSettings() {
        AiProviderSettings settings = AiProviderSettings.from(properties);

        override(settings.getEmbedding(), KEY_EMBEDDING_BASE_URL, KEY_EMBEDDING_API_KEY,
                KEY_EMBEDDING_MODEL, properties.getEmbedding());
        override(settings.getImageEmbedding(), KEY_IMAGE_BASE_URL, KEY_IMAGE_API_KEY,
                KEY_IMAGE_MODEL, properties.getImageEmbedding());
        override(settings.getLlm(), KEY_LLM_BASE_URL, KEY_LLM_API_KEY,
                KEY_LLM_MODEL, properties.getLlm());

        overrideIfPresent(KEY_VECTOR_STORE, settings::setVectorStoreOverride);
        overrideIfPresent(KEY_MILVUS_HOST, settings::setMilvusHostOverride);
        overrideIfPresent(KEY_MILVUS_PORT, v -> settings.setMilvusPortOverride(Integer.parseInt(v.trim())));
        overrideIfPresent(KEY_MILVUS_COLLECTION_TEXT, settings::setMilvusCollectionTextOverride);
        overrideIfPresent(KEY_MILVUS_COLLECTION_IMAGE, settings::setMilvusCollectionImageOverride);

        return settings;
    }

    private void overrideIfPresent(String key, Consumer<String> setter) {
        configRepository.findValue(key)
                .filter(v -> !v.isBlank())
                .ifPresent(setter);
    }

    private void override(AiProviderSettings.Endpoint endpoint, String baseUrlKey, String apiKeyKey,
                          String modelKey, AiShoppingProperties.Endpoint defaults) {
        endpoint.baseUrl = configRepository.findValue(baseUrlKey)
                .filter(v -> !v.isBlank()).orElse(defaults.getBaseUrl());
        endpoint.apiKey = configRepository.findValue(apiKeyKey)
                .filter(v -> !v.isBlank()).orElse(defaults.getApiKey());
        endpoint.model = configRepository.findValue(modelKey)
                .filter(v -> !v.isBlank()).orElse(defaults.getModel());
        endpoint.timeoutMs = defaults.getTimeoutMs();
        endpoint.maxRetries = defaults.getMaxRetries();
    }
}
