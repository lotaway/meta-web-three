package com.metawebthree.recommendation.infrastructure.aishopping.vector;

import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class InMemoryVectorStoreTest {

    private InMemoryVectorStore store;

    @BeforeEach
    void setUp() {
        store = new InMemoryVectorStore();
        store.ensureCollection("product_text", 4);
        store.upsert("product_text", List.of(
                new VectorRecord(1, 101, new float[]{1, 0, 0, 0}, Map.of()),
                new VectorRecord(2, 102, new float[]{0, 1, 0, 0}, Map.of()),
                new VectorRecord(3, 103, new float[]{0, 0, 1, 0}, Map.of())));
    }

    @Test
    void search_shouldReturnHitsSortedByScoreDesc() {
        List<VectorHit> hits = store.search("product_text", new float[]{1, 0, 0, 0}, 2);

        assertEquals(2, hits.size());
        assertEquals(101, hits.get(0).getProductId());
        assertEquals(102, hits.get(1).getProductId());
        assertEquals(1.0f, hits.get(0).getScore(), 0.0001);
    }

    @Test
    void search_shouldRespectTopK() {
        List<VectorHit> hits = store.search("product_text", new float[]{1, 1, 1, 1}, 1);

        assertEquals(1, hits.size());
    }

    @Test
    void upsert_shouldReplaceExistingRecord() {
        store.upsert("product_text", List.of(
                new VectorRecord(1, 999, new float[]{0, 0, 0, 1}, Map.of())));

        List<VectorHit> hits = store.search("product_text", new float[]{0, 0, 0, 1}, 1);

        assertEquals(1, hits.size());
        assertEquals(999, hits.get(0).getProductId());
    }

    @Test
    void count_shouldReturnNumberOfRecords() {
        assertEquals(3, store.count("product_text"));
        assertEquals(0, store.count("missing"));
    }
}
