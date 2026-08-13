package com.metawebthree.recommendation.infrastructure.aishopping.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * Default AI shopping configuration from the {@code ai-shopping.*} prefix in
 * application.yml. Same-named rows in the ai_shopping_config table override
 * these defaults.
 */
@ConfigurationProperties(prefix = "ai-shopping")
public class AiShoppingProperties {

    /** Vector storage backend: milvus | memory */
    private String vectorStore = "memory";
    private int embeddingDim = 1024;
    private int defaultTopK = 20;
    private boolean enabled = true;

    private Endpoint embedding = new Endpoint();
    private Endpoint imageEmbedding = new Endpoint();
    private Endpoint llm = new Endpoint();
    private Milvus milvus = new Milvus();

    public static class Endpoint {
        private String baseUrl = "";
        private String apiKey = "";
        private String model = "";
        /** Custom request path; defaults to the OpenAI-compatible protocol path */
        private String path = "";
        private int timeoutMs = 15000;
        private int maxRetries = 2;

        public String getBaseUrl() {
            return baseUrl;
        }

        public void setBaseUrl(String baseUrl) {
            this.baseUrl = baseUrl;
        }

        public String getApiKey() {
            return apiKey;
        }

        public void setApiKey(String apiKey) {
            this.apiKey = apiKey;
        }

        public String getModel() {
            return model;
        }

        public void setModel(String model) {
            this.model = model;
        }

        public String getPath() {
            return path;
        }

        public void setPath(String path) {
            this.path = path;
        }

        public int getTimeoutMs() {
            return timeoutMs;
        }

        public void setTimeoutMs(int timeoutMs) {
            this.timeoutMs = timeoutMs;
        }

        public int getMaxRetries() {
            return maxRetries;
        }

        public void setMaxRetries(int maxRetries) {
            this.maxRetries = maxRetries;
        }
    }

    public static class Milvus {
        private String host = "localhost";
        private int port = 19530;
        private String collectionText = "product_text";
        private String collectionImage = "product_image";
        private String apiToken = "";

        public String getHost() {
            return host;
        }

        public void setHost(String host) {
            this.host = host;
        }

        public int getPort() {
            return port;
        }

        public void setPort(int port) {
            this.port = port;
        }

        public String getCollectionText() {
            return collectionText;
        }

        public void setCollectionText(String collectionText) {
            this.collectionText = collectionText;
        }

        public String getCollectionImage() {
            return collectionImage;
        }

        public void setCollectionImage(String collectionImage) {
            this.collectionImage = collectionImage;
        }

        public String getApiToken() {
            return apiToken;
        }

        public void setApiToken(String apiToken) {
            this.apiToken = apiToken;
        }
    }

    public String getVectorStore() {
        return vectorStore;
    }

    public void setVectorStore(String vectorStore) {
        this.vectorStore = vectorStore;
    }

    public int getEmbeddingDim() {
        return embeddingDim;
    }

    public void setEmbeddingDim(int embeddingDim) {
        this.embeddingDim = embeddingDim;
    }

    public int getDefaultTopK() {
        return defaultTopK;
    }

    public void setDefaultTopK(int defaultTopK) {
        this.defaultTopK = defaultTopK;
    }

    public boolean isEnabled() {
        return enabled;
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }

    public Endpoint getEmbedding() {
        return embedding;
    }

    public void setEmbedding(Endpoint embedding) {
        this.embedding = embedding;
    }

    public Endpoint getImageEmbedding() {
        return imageEmbedding;
    }

    public void setImageEmbedding(Endpoint imageEmbedding) {
        this.imageEmbedding = imageEmbedding;
    }

    public Endpoint getLlm() {
        return llm;
    }

    public void setLlm(Endpoint llm) {
        this.llm = llm;
    }

    public Milvus getMilvus() {
        return milvus;
    }

    public void setMilvus(Milvus milvus) {
        this.milvus = milvus;
    }
}
