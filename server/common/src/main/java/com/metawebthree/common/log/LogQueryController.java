package com.metawebthree.common.log;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

/**
 * REST controller for log querying and aggregation
 */
@RestController
@RequestMapping("/logs")
public class LogQueryController {
    
    @Autowired
    private LogQueryService logQueryService;
    
    /**
     * Query logs with multi-dimensional filtering
     */
    @PostMapping("/query")
    public ResponseEntity<?> queryLogs(@RequestBody LogQueryRequest request) {
        List<LogEntry> logs = logQueryService.queryLogs(request);
        return ResponseEntity.ok(logs);
    }
    
    /**
     * Query logs by time range (convenience endpoint)
     */
    @GetMapping("/query")
    public ResponseEntity<?> queryLogs(
            @RequestParam(value = "startTime", required = false) Long startTime,
            @RequestParam(value = "endTime", required = false) Long endTime,
            @RequestParam(value = "level", required = false) String level,
            @RequestParam(value = "keyword", required = false) String keyword,
            @RequestParam(value = "loggerName", required = false) String loggerName,
            @RequestParam(value = "limit", defaultValue = "100") int limit) {
        
        LogQueryRequest request = new LogQueryRequest();
        request.setStartTime(startTime);
        request.setEndTime(endTime);
        request.setLevel(level);
        request.setKeyword(keyword);
        request.setLoggerName(loggerName);
        request.setLimit(limit);
        
        List<LogEntry> logs = logQueryService.queryLogs(request);
        return ResponseEntity.ok(logs);
    }
    
    /**
     * Get log statistics (aggregation by level, time, etc.)
     */
    @GetMapping("/statistics")
    public ResponseEntity<?> getLogStatistics(
            @RequestParam(value = "startTime", required = false) Long startTime,
            @RequestParam(value = "endTime", required = false) Long endTime) {
        
        LogStatistics stats = logQueryService.getLogStatistics(startTime, endTime);
        return ResponseEntity.ok(stats);
    }
    
    /**
     * Get log level distribution
     */
    @GetMapping("/statistics/levels")
    public ResponseEntity<?> getLogLevelDistribution(
            @RequestParam(value = "startTime", required = false) Long startTime,
            @RequestParam(value = "endTime", required = false) Long endTime) {
        
        LogStatistics stats = logQueryService.getLogStatistics(startTime, endTime);
        Map<String, Long> levelCounts = stats.getLevelCounts();
        
        return ResponseEntity.ok(levelCounts);
    }
    
    /**
     * Search logs by keyword (convenience endpoint)
     */
    @GetMapping("/search")
    public ResponseEntity<?> searchLogs(
            @RequestParam("keyword") String keyword,
            @RequestParam(value = "limit", defaultValue = "100") int limit) {
        
        LogQueryRequest request = new LogQueryRequest();
        request.setKeyword(keyword);
        request.setLimit(limit);
        
        List<LogEntry> logs = logQueryService.queryLogs(request);
        return ResponseEntity.ok(logs);
    }
}
