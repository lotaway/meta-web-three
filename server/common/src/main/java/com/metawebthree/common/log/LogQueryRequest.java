package com.metawebthree.common.log;

/**
 * Request parameters for log querying
 */
public class LogQueryRequest {
    private Long startTime;
    private Long endTime;
    private String level;
    private String keyword;
    private String loggerName;
    private int limit = 100;
    
    public Long getStartTime() {
        return startTime;
    }
    
    public void setStartTime(Long startTime) {
        this.startTime = startTime;
    }
    
    public Long getEndTime() {
        return endTime;
    }
    
    public void setEndTime(Long endTime) {
        this.endTime = endTime;
    }
    
    public String getLevel() {
        return level;
    }
    
    public void setLevel(String level) {
        this.level = level;
    }
    
    public String getKeyword() {
        return keyword;
    }
    
    public void setKeyword(String keyword) {
        this.keyword = keyword;
    }
    
    public String getLoggerName() {
        return loggerName;
    }
    
    public void setLoggerName(String loggerName) {
        this.loggerName = loggerName;
    }
    
    public int getLimit() {
        return limit;
    }
    
    public void setLimit(int limit) {
        this.limit = limit;
    }
}
