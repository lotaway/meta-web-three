package com.metawebthree.common.utils;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;

public class DataBucketMap<Data> {
    private final Map<Data, Integer> dataMap = new ConcurrentHashMap<>();
    private final AtomicInteger pos = new AtomicInteger(0);

    public Integer add(Data data, Integer weight) {
        return dataMap.put(data, weight);
    }

    public Integer remove(Data data) {
        return dataMap.remove(data);
    }

    public Data choose(BucketChooseMethod bucketChooseMethod) {
        return switch (bucketChooseMethod) {
            case BY_ROUND_ROBIN -> chooseByRoundRobin();
            case BY_RANDOM -> chooseByRandom();
            case BY_WEIGHT_ROUND_ROBIN -> chooseByWeightRoundBin();
            case BY_WEIGHT_RANDOM -> chooseByWeightRandom();
            default -> chooseByRoundRobin();
        };
    }

    public Data chooseByRoundRobin() {
        List<Data> snapshot = new ArrayList<>(dataMap.keySet());
        if (snapshot.isEmpty()) {
            return null;
        }
        int index = Math.floorMod(pos.getAndIncrement(), snapshot.size());
        return snapshot.get(index);
    }

    public Data chooseByRandom() {
        List<Data> snapshot = new ArrayList<>(dataMap.keySet());
        if (snapshot.isEmpty()) {
            return null;
        }
        int chosenPos = ThreadLocalRandom.current().nextInt(snapshot.size());
        return snapshot.get(chosenPos);
    }

    public Data chooseByWeightRoundBin() {
        List<Data> weightedSnapshot = buildWeightedSnapshot();
        if (weightedSnapshot.isEmpty()) {
            return null;
        }
        int index = Math.floorMod(pos.getAndIncrement(), weightedSnapshot.size());
        return weightedSnapshot.get(index);
    }

    public Data chooseByWeightRandom() {
        List<Data> weightedSnapshot = buildWeightedSnapshot();
        if (weightedSnapshot.isEmpty()) {
            return null;
        }
        int chosenPos = ThreadLocalRandom.current().nextInt(weightedSnapshot.size());
        return weightedSnapshot.get(chosenPos);
    }

    private List<Data> buildWeightedSnapshot() {
        List<Data> weightedSnapshot = new ArrayList<>();
        for (Map.Entry<Data, Integer> entry : dataMap.entrySet()) {
            Integer weight = entry.getValue();
            if (weight == null || weight <= 0) {
                continue;
            }
            for (int i = 0; i < weight; i++) {
                weightedSnapshot.add(entry.getKey());
            }
        }
        return weightedSnapshot;
    }

    // @TODO Least Connections）
    public Data chooseByLeastConnection() {
        return null;
    }
}
