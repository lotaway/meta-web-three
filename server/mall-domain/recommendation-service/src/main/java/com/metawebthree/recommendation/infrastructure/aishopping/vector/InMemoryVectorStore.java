package com.metawebthree.recommendation.infrastructure.aishopping.vector;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * In-memory vector store (fallback when Milvus is unavailable, for unit tests,
 * or for small catalogs). Uses brute-force cosine similarity.
 */
public class InMemoryVectorStore implements VectorStore {

    private final Map<String, List<VectorRecord>> collections = new ConcurrentHashMap<>();
    private final AtomicLong idSeq = new AtomicLong(1);

    @Override
    public String name() {
        return "memory";
    }

    @Override
    public void ensureCollection(String collection, int dim) {
        collections.computeIfAbsent(collection, k -> new ArrayList<>());
    }

    @Override
    public synchronized void upsert(String collection, List<VectorRecord> records) {
        List<VectorRecord> bucket = collections.computeIfAbsent(collection, k -> new ArrayList<>());
        bucket.removeIf(existing -> records.stream().anyMatch(r -> r.getId() == existing.getId()));
        bucket.addAll(records);
    }

    @Override
    public List<VectorHit> search(String collection, float[] query, int topK) {
        List<VectorRecord> bucket = collections.getOrDefault(collection, List.of());
        return bucket.stream()
                .map(record -> new VectorHit(
                        record.getId(), record.getProductId(), cosine(query, record.getVector()), record.getMetadata()))
                .sorted((a, b) -> Float.compare(b.getScore(), a.getScore()))
                .limit(topK)
                .toList();
    }

    @Override
    public long count(String collection) {
        return collections.getOrDefault(collection, List.of()).size();
    }

    @Override
    public void drop(String collection) {
        collections.remove(collection);
    }

    private float cosine(float[] a, float[] b) {
        double dot = 0.0, normA = 0.0, normB = 0.0;
        int len = Math.min(a.length, b.length);
        for (int i = 0; i < len; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        if (normA == 0.0 || normB == 0.0) {
            return 0.0f;
        }
        return (float) (dot / (Math.sqrt(normA) * Math.sqrt(normB)));
    }
}
