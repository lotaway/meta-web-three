package com.metawebthree.recommendation.infrastructure.aishopping.vector;

import java.util.List;

/**
 * Vector storage abstraction. Implementations: MilvusVectorStore /
 * InMemoryVectorStore.
 */
public interface VectorStore {

    String name();

    void ensureCollection(String collection, int dim);

    void upsert(String collection, List<VectorRecord> records);

    List<VectorHit> search(String collection, float[] query, int topK);

    long count(String collection);

    void drop(String collection);
}
