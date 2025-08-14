package com.metawebthree.common.utils;

import java.sql.Timestamp;
import java.time.LocalDateTime;
import java.util.concurrent.atomic.AtomicInteger;

import com.baomidou.mybatisplus.core.toolkit.IdWorker;

public class GeneIdWorker extends IdWorker {

    protected AtomicInteger counter = new AtomicInteger(0);

    public long getId(Long userId) {
        int shardGene = Math.abs(userId.hashCode()) % 10000;
        Timestamp timePart = Timestamp.valueOf(LocalDateTime.now());
        int seqPart = counter.getAndIncrement() % 10000;
        return (long) (shardGene * 1000000000000L + timePart.getTime() * 10000 + seqPart);
    }

    public int extractShardGene(long orderNo) {
        return Integer.parseInt(String.valueOf(orderNo).substring(0, 4));
    }
}
