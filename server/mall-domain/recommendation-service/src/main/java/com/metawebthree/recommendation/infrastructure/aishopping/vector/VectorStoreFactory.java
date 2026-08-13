package com.metawebthree.recommendation.infrastructure.aishopping.vector;

import com.metawebthree.recommendation.application.aishopping.AiProviderConfig;
import com.metawebthree.recommendation.application.aishopping.AiProviderSettings;
import org.springframework.stereotype.Component;

/** Selects the vector store implementation based on the active configuration. */
@Component
public class VectorStoreFactory {

    private final AiProviderConfig providerConfig;
    private volatile VectorStore cachedStore;
    private volatile String cachedMode;

    public VectorStoreFactory(AiProviderConfig providerConfig) {
        this.providerConfig = providerConfig;
    }

    public VectorStore getStore() {
        AiProviderSettings settings = providerConfig.getSettings();
        String mode = settings.getVectorStore() == null ? "memory" : settings.getVectorStore().toLowerCase();
        if (cachedStore == null || !mode.equals(cachedMode)) {
            synchronized (this) {
                if (cachedStore == null || !mode.equals(cachedMode)) {
                    cachedMode = mode;
                    cachedStore = create(mode, settings);
                }
            }
        }
        return cachedStore;
    }

    private VectorStore create(String mode, AiProviderSettings settings) {
        if ("milvus".equals(mode)) {
            return new MilvusVectorStore(
                    settings.getMilvusHost(), settings.getMilvusPort(), settings.getMilvusApiToken());
        }
        return new InMemoryVectorStore();
    }
}
