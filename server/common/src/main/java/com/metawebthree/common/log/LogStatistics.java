package com.metawebthree.common.log;

import java.util.HashMap;
import java.util.Map;

/**
 * Log statistics for aggregation
 */
public class LogStatistics {
    private long total;
    private Map<String, Long> levelCounts = new HashMap<>();
    
    public long getTotal() {
        return total;
    }
    
    public void setTotal(long total) {
        this.total = total;
    }
    
    public void incrementTotal() {
        this.total++;
    }
    
    public Map<String, Long> getLevelCounts() {
        return levelCounts;
    }
    
    public void setLevelCounts(Map<String, Long> levelCounts) {
        this.levelCounts = levelCounts;
    }
    
    public void incrementLevel(String level) {
        levelCounts.merge(level, 1L, Long::sum);
    }
    
    public long getErrorCount() {
        return levelCounts.getOrDefault("ERROR", 0L);
    }
    
    public long getWarnCount() {
        return levelCounts.getOrDefault("WARN", 0L);
    }
    
    public long getInfoCount() {
        return levelCounts.getOrDefault("INFO", 0L);
    }
    
    public long getDebugCount() {
        return levelCounts.getOrDefault("DEBUG", 0L);
    }
}
