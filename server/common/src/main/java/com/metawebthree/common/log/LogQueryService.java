package com.metawebthree.common.log;

import org.rocksdb.Options;
import org.rocksdb.RocksDB;
import org.rocksdb.RocksDBException;
import org.rocksdb.RocksIterator;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.StringJoiner;

/**
 * Service for querying logs from RocksDB with multi-dimensional filtering
 */
@Service
public class LogQueryService {

    private static final Logger logger = LoggerFactory.getLogger(LogQueryService.class);

    @Value("${logging.data-path:logs/rocksdb-logs}")
    private String logPath;
    
    private RocksDB db;
    private String serviceName;
    private static final String SEQ_KEY = "SystemOut";
    
    @PostConstruct
    public void init() {
        Properties properties = System.getProperties();
        serviceName = System.getProperty("APPLICATION_NAME");
        if (serviceName == null) {
            serviceName = System.getProperty("spring.boot.project.name", "default");
        }
        
        RocksDB.loadLibrary();
        try (final Options options = new Options().setCreateIfMissing(true)) {
            var sj = new StringJoiner("/");
            sj.add(logPath).add(serviceName);
            String path = new File(sj.toString()).getAbsolutePath();
            
            File dir = new File(path);
            if (!dir.exists()) {
                dir.mkdirs();
            }
            
            db = RocksDB.open(options, path);
        } catch (RocksDBException e) {
            logger.error("Failed to open RocksDB for log querying", e);
        }
    }
    
    /**
     * Query logs with multi-dimensional filtering
     */
    public List<LogEntry> queryLogs(LogQueryRequest request) {
        List<LogEntry> results = new ArrayList<>();
        
        if (db == null) {
            return results;
        }
        
        try (final RocksIterator iterator = db.newIterator()) {
            iterator.seekToFirst();
            
            int count = 0;
            while (iterator.isValid() && count < request.getLimit()) {
                String key = new String(iterator.key());
                String value = new String(iterator.value());
                
                // Parse log entry
                LogEntry entry = parseLogEntry(key, value);
                
                if (entry != null && matchesFilter(entry, request)) {
                    results.add(entry);
                    count++;
                }
                
                iterator.next();
            }
        } catch (Exception e) {
            logger.error("Error querying logs", e);
        }
        
        return results;
    }
    
    /**
     * Parse log entry from key-value pair
     */
    private LogEntry parseLogEntry(String key, String value) {
        try {
            String[] parts = value.split("\\|", 5);
            if (parts.length >= 5) {
                LogEntry entry = new LogEntry();
                entry.setTimestamp(Long.parseLong(parts[0]));
                entry.setLevel(parts[1]);
                entry.setThread(parts[2]);
                entry.setLoggerName(parts[3]);
                entry.setMessage(parts[4]);
                entry.setKey(key);
                return entry;
            }
        } catch (Exception e) {
            // Skip malformed entries
        }
        return null;
    }
    
    /**
     * Check if log entry matches the filter criteria
     */
    private boolean matchesFilter(LogEntry entry, LogQueryRequest request) {
        // Filter by time range
        if (request.getStartTime() != null && entry.getTimestamp() < request.getStartTime()) {
            return false;
        }
        if (request.getEndTime() != null && entry.getTimestamp() > request.getEndTime()) {
            return false;
        }
        
        // Filter by log level
        if (request.getLevel() != null && !request.getLevel().isEmpty()) {
            if (!entry.getLevel().equalsIgnoreCase(request.getLevel())) {
                return false;
            }
        }
        
        // Filter by keyword
        if (request.getKeyword() != null && !request.getKeyword().isEmpty()) {
            if (!entry.getMessage().contains(request.getKeyword()) &&
                !entry.getLoggerName().contains(request.getKeyword())) {
                return false;
            }
        }
        
        // Filter by logger name (service/component)
        if (request.getLoggerName() != null && !request.getLoggerName().isEmpty()) {
            if (!entry.getLoggerName().contains(request.getLoggerName())) {
                return false;
            }
        }
        
        return true;
    }
    
    /**
     * Get log statistics (aggregation by level, time, etc.)
     */
    public LogStatistics getLogStatistics(Long startTime, Long endTime) {
        LogStatistics stats = new LogStatistics();
        
        if (db == null) {
            return stats;
        }
        
        try (final RocksIterator iterator = db.newIterator()) {
            iterator.seekToFirst();
            
            while (iterator.isValid()) {
                String value = new String(iterator.value());
                LogEntry entry = parseLogEntry(new String(iterator.key()), value);
                
                if (entry != null) {
                    // Filter by time range if specified
                    if (startTime != null && entry.getTimestamp() < startTime) {
                        iterator.next();
                        continue;
                    }
                    if (endTime != null && entry.getTimestamp() > endTime) {
                        iterator.next();
                        continue;
                    }
                    
                    stats.incrementTotal();
                    stats.incrementLevel(entry.getLevel());
                }
                
                iterator.next();
            }
        } catch (Exception e) {
            logger.error("Error getting log statistics", e);
        }
        
        return stats;
    }
    
    @PreDestroy
    public void cleanup() {
        if (db != null) {
            db.close();
        }
    }
}
