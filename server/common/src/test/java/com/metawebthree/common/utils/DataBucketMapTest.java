package com.metawebthree.common.utils;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.junit.jupiter.api.Test;

class DataBucketMapTest {

    @Test
    void chooseByRoundRobin_shouldReturnInsertedElement() {
        DataBucketMap<String> bucketMap = new DataBucketMap<>();
        bucketMap.add("A", 1);

        String chosen = bucketMap.chooseByRoundRobin();

        assertEquals("A", chosen);
    }

    @Test
    void chooseByWeightRoundBin_shouldPreferWeightedData() {
        DataBucketMap<String> bucketMap = new DataBucketMap<>();
        bucketMap.add("A", 0);
        bucketMap.add("B", 3);

        for (int i = 0; i < 5; i++) {
            assertEquals("B", bucketMap.chooseByWeightRoundBin());
        }
    }

    @Test
    void chooseByRoundRobin_shouldBeThreadSafe() throws Exception {
        DataBucketMap<String> bucketMap = new DataBucketMap<>();
        bucketMap.add("A", 1);
        bucketMap.add("B", 1);

        ExecutorService executor = Executors.newFixedThreadPool(8);
        List<Callable<String>> tasks = new ArrayList<>();
        for (int i = 0; i < 200; i++) {
            tasks.add(() -> bucketMap.chooseByRoundRobin());
        }

        List<Future<String>> futures = executor.invokeAll(tasks);
        executor.shutdown();

        for (Future<String> future : futures) {
            String chosen = future.get();
            assertNotNull(chosen);
            assertTrue(Set.of("A", "B").contains(chosen));
        }
    }
}
