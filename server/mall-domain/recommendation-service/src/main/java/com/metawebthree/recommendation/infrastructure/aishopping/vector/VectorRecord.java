package com.metawebthree.recommendation.infrastructure.aishopping.vector;

import java.util.Map;

/** Vector store record. */
public class VectorRecord {

    private final long id;
    private final long productId;
    private final float[] vector;
    private final Map<String, Object> metadata;

    public VectorRecord(long id, long productId, float[] vector, Map<String, Object> metadata) {
        this.id = id;
        this.productId = productId;
        this.vector = vector;
        this.metadata = metadata;
    }

    public long getId() {
        return id;
    }

    public long getProductId() {
        return productId;
    }

    public float[] getVector() {
        return vector;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }
}
