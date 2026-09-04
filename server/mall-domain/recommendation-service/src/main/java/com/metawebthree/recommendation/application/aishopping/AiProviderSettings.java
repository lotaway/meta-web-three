package com.metawebthree.recommendation.application.aishopping;

import com.metawebthree.recommendation.infrastructure.aishopping.config.AiShoppingProperties;

/** Effective configuration after merging application.yml defaults and DB overrides. */
public class AiProviderSettings {

    public static class Endpoint {
        public String baseUrl;
        public String apiKey;
        public String model;
        public String path;
        public int timeoutMs;
        public int maxRetries;

        public boolean isConfigured() {
            return baseUrl != null && !baseUrl.isBlank();
        }

        public String resolvePath(String defaultPath) {
            return path != null && !path.isBlank() ? path : defaultPath;
        }
    }

    private final Endpoint embedding;
    private final Endpoint imageEmbedding;
    private final Endpoint llm;
    private String vectorStore;
    private String milvusHost;
    private int milvusPort;
    private String milvusCollectionText;
    private String milvusCollectionImage;
    private final String milvusApiToken;
    private final int embeddingDim;
    private final int defaultTopK;

    public AiProviderSettings(
            Endpoint embedding, Endpoint imageEmbedding, Endpoint llm,
            String vectorStore, String milvusHost, int milvusPort,
            String milvusCollectionText, String milvusCollectionImage, String milvusApiToken,
            int embeddingDim, int defaultTopK) {
        this.embedding = embedding;
        this.imageEmbedding = imageEmbedding;
        this.llm = llm;
        this.vectorStore = vectorStore;
        this.milvusHost = milvusHost;
        this.milvusPort = milvusPort;
        this.milvusCollectionText = milvusCollectionText;
        this.milvusCollectionImage = milvusCollectionImage;
        this.milvusApiToken = milvusApiToken;
        this.embeddingDim = embeddingDim;
        this.defaultTopK = defaultTopK;
    }

    public static AiProviderSettings from(AiShoppingProperties props) {
        return new AiProviderSettings(
                copy(props.getEmbedding()),
                copy(props.getImageEmbedding()),
                copy(props.getLlm()),
                props.getVectorStore(),
                props.getMilvus().getHost(),
                props.getMilvus().getPort(),
                props.getMilvus().getCollectionText(),
                props.getMilvus().getCollectionImage(),
                props.getMilvus().getApiToken(),
                props.getEmbeddingDim(),
                props.getDefaultTopK());
    }

    private static Endpoint copy(AiShoppingProperties.Endpoint source) {
        Endpoint target = new Endpoint();
        target.baseUrl = source.getBaseUrl();
        target.apiKey = source.getApiKey();
        target.model = source.getModel();
        target.path = source.getPath();
        target.timeoutMs = source.getTimeoutMs();
        target.maxRetries = source.getMaxRetries();
        return target;
    }

    public Endpoint getEmbedding() {
        return embedding;
    }

    public Endpoint getImageEmbedding() {
        return imageEmbedding;
    }

    public Endpoint getLlm() {
        return llm;
    }

    public String getVectorStore() {
        return vectorStore;
    }

    public String getMilvusHost() {
        return milvusHost;
    }

    public int getMilvusPort() {
        return milvusPort;
    }

    public String getMilvusCollectionText() {
        return milvusCollectionText;
    }

    public String getMilvusCollectionImage() {
        return milvusCollectionImage;
    }

    public String getMilvusApiToken() {
        return milvusApiToken;
    }

    public int getEmbeddingDim() {
        return embeddingDim;
    }

    public int getDefaultTopK() {
        return defaultTopK;
    }

    public void setVectorStoreOverride(String value) {
        this.vectorStore = value;
    }

    public void setMilvusHostOverride(String value) {
        this.milvusHost = value;
    }

    public void setMilvusPortOverride(int value) {
        this.milvusPort = value;
    }

    public void setMilvusCollectionTextOverride(String value) {
        this.milvusCollectionText = value;
    }

    public void setMilvusCollectionImageOverride(String value) {
        this.milvusCollectionImage = value;
    }

    public AiProviderSettings withOverrides(AiShoppingProperties defaults) {
        fill(embedding, defaults.getEmbedding());
        fill(imageEmbedding, defaults.getImageEmbedding());
        fill(llm, defaults.getLlm());
        return this;
    }

    private void fill(Endpoint target, AiShoppingProperties.Endpoint defaults) {
        if (isBlank(target.baseUrl)) {
            target.baseUrl = defaults.getBaseUrl();
        }
        if (isBlank(target.apiKey)) {
            target.apiKey = defaults.getApiKey();
        }
        if (isBlank(target.model)) {
            target.model = defaults.getModel();
        }
    }

    private static boolean isBlank(String value) {
        return value == null || value.isBlank();
    }
}
