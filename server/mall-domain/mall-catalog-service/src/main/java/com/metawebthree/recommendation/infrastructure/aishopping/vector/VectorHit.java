package com.metawebthree.recommendation.infrastructure.aishopping.vector;

import java.util.Map;

/** Vector search hit. */
public class VectorHit {

    private final long id;
    private final long productId;
    private final float score;
    private final Map<String, Object> metadata;

    public VectorHit(long id, long productId, float score, Map<String, Object> metadata) {
        this.id = id;
        this.productId = productId;
        this.score = score;
        this.metadata = metadata;
    }

    public long getId() {
        return id;
    }

    public long getProductId() {
        return productId;
    }

    public float getScore() {
        return score;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }
}
