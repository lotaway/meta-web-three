package com.metawebthree.order.application;

import com.metawebthree.order.infrastructure.persistence.mapper.AdminOrderMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Admin order query service for statistics
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class AdminOrderQueryService {

    private final AdminOrderMapper adminOrderMapper;

    /**
     * Get order status distribution
     * Status: 0->pending, 1->processed, 2->shipped, 3->completed, 4->closed
     * @return map of status -> count
     */
    public Map<String, Long> getOrderStatusDistribution() {
        try {
            List<Map<String, Object>> results = adminOrderMapper.selectOrderCountGroupByStatus();
            Map<String, Long> distribution = new HashMap<>();
            
            Map<Integer, String> statusNames = new HashMap<>();
            statusNames.put(0, "PENDING");
            statusNames.put(1, "PROCESSED");
            statusNames.put(2, "SHIPPED");
            statusNames.put(3, "COMPLETED");
            statusNames.put(4, "CLOSED");
            
            for (Map<String, Object> row : results) {
                Object statusObj = row.get("status");
                Object countObj = row.get("count");
                
                if (statusObj != null && countObj != null) {
                    Integer status = ((Number) statusObj).intValue();
                    Long count = ((Number) countObj).longValue();
                    String statusName = statusNames.getOrDefault(status, "UNKNOWN");
                    distribution.put(statusName, count);
                }
            }
            
            return distribution;
        } catch (Exception e) {
            log.error("Failed to get order status distribution", e);
            return new HashMap<>();
        }
    }

    /**
     * Get pending orders count (status in 0,1,2)
     * Status: 0->pending, 1->processed, 2->shipped, 3->completed, 4->closed
     * @return count of pending orders
     */
    public Long getPendingOrdersCount() {
        try {
            Long count = adminOrderMapper.selectPendingOrdersCount();
            return count != null ? count : 0L;
        } catch (Exception e) {
            log.error("Failed to get pending orders count", e);
            return 0L;
        }
    }
    
    /**
     * Get pending payments count (orders with status 0, waiting for payment)
     * @return count of orders awaiting payment
     */
    public Long getPendingPaymentsCount() {
        try {
            // Status 0 = pending payment (order created but not paid)
            Long count = adminOrderMapper.selectPendingPaymentsCount();
            return count != null ? count : 0L;
        } catch (Exception e) {
            log.error("Failed to get pending payments count", e);
            return 0L;
        }
    }
    
    /**
     * Get hot products based on sales
     * @param limit number of products to return
     * @return list of product info with sales data
     */
    public List<Map<String, Object>> getHotProducts(int limit) {
        try {
            List<Map<String, Object>> results = adminOrderMapper.selectHotProducts(limit);
            return results != null ? results : List.of();
        } catch (Exception e) {
            log.error("Failed to get hot products", e);
            return List.of();
        }
    }
    
    /**
     * Get sales by hour for today
     * @return list of hourly sales data
     */
    public List<Map<String, Object>> getSalesByHourToday() {
        try {
            List<Map<String, Object>> results = adminOrderMapper.selectSalesByHourToday();
            return results != null ? results : List.of();
        } catch (Exception e) {
            log.error("Failed to get sales by hour today", e);
            return List.of();
        }
    }
}