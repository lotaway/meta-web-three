package com.metawebthree.common.log;

/**
 * Log entry data structure
 */
public class LogEntry {
    private String key;
    private long timestamp;
    private String level;
    private String thread;
    private String loggerName;
    private String message;
    
    public String getKey() {
        return key;
    }
    
    public void setKey(String key) {
        this.key = key;
    }
    
    public long getTimestamp() {
        return timestamp;
    }
    
    public void setTimestamp(long timestamp) {
        this.timestamp = timestamp;
    }
    
    public String getLevel() {
        return level;
    }
    
    public void setLevel(String level) {
        this.level = level;
    }
    
    public String getThread() {
        return thread;
    }
    
    public void setThread(String thread) {
        this.thread = thread;
    }
    
    public String getLoggerName() {
        return loggerName;
    }
    
    public void setLoggerName(String loggerName) {
        this.loggerName = loggerName;
    }
    
    public String getMessage() {
        return message;
    }
    
    public void setMessage(String message) {
        this.message = message;
    }
}
