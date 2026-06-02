package com.metawebthree.dataanalysis.infrastructure.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.common.generated.rpc.*;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Client for calling order-service via REST API
 */
@Slf4j
@Component
public class OrderClient {

    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper;

    @Value(\"${services.order-service.url:http://localhost:8082}\")
    private String orderServiceUrl;

    public OrderClient() {
        this.restTemplate = new RestTemplate();
        this.objectMapper = new ObjectMapper();
    }

    /**
     * Get order status distribution counts
     * @return Map of status -> count
     */
    public Map<String, Long> getOrderStatusDistribution() {
        try {
            String url = orderServiceUrl + \"/api/admin/order/statistics/status-distribution\";
            ResponseEntity<Map> response = restTemplate.getForEntity(url, Map.class);
            if (response.getBody() != null && response.getBody().containsKey(\"data\")) {
                @SuppressWarnings(\"unchecked\")
                Map<String, Long> data = (Map<String, Long>) response.getBody().get(\"data\");
                return data != null ? data : new HashMap<>();
            }
        } catch (Exception e) {
            log.warn(\"Failed to get order status distribution: {}\", e.getMessage());
        }
        return new HashMap<>();
    }

    /**
     * Get pending orders count
     * @return count of pending orders
     */
    public Long getPendingOrdersCount() {
        try {
            String url = orderServiceUrl + \"/api/admin/order/statistics/pending-count\";
            ResponseEntity<Map> response = restTemplate.getForEntity(url, Map.class);
            if (response.getBody() != null && response.getBody().containsKey(\"data\")) {
                Object data = response.getBody().get(\"data\");
                if (data instanceof Number) {
                    return ((Number) data).longValue();
                }
            }
        } catch (Exception e) {
            log.warn(\"Failed to get pending orders count: {}\", e.getMessage());
        }
        return 0L;
    }

    /**
     * Get pending payments count (orders awaiting payment)
     * @return count of orders waiting for payment
     */
    public Long getPendingPaymentsCount() {
        try {
            String url = orderServiceUrl + \"/api/admin/order/statistics/pending-payments-count\";
            ResponseEntity<Map> response = restTemplate.getForEntity(url, Map.class);
            if (response.getBody() != null && response.getBody().containsKey(\"data\")) {
                Object data = response.getBody().get(\"data\");
                if (data instanceof Number) {
                    return ((Number) data).longValue();
                }
            }
        } catch (Exception e) {
            log.warn(\"Failed to get pending payments count: {}\", e.getMessage());
        }
        return 0L;
    }

    /**
     * Get hot products based on sales
     * @param limit number of products to return
     * @return list of hot product info
     */
    public List<HotProductInfo> getHotProducts(int limit) {
        // Placeholder - would query from order-service or product-service
        // This requires aggregation of order items by product
        return new ArrayList<>();
    }

    /**
     * Get sales by hour for today
     * @return list of hourly sales data
     */
    public List<SalesByHourInfo> getSalesByHourToday() {
        // Placeholder - would query from order-service
        // Aggregating order data by hour
        return new ArrayList<>();
    }

    /**
     * Hot product info DTO
     */
    public static class HotProductInfo {
        private Long productId;
        private String productName;
        private Long salesCount;
        private Long salesAmount;

        public Long getProductId() { return productId; }
        public void setProductId(Long productId) { this.productId = productId; }
        public String getProductName() { return productName; }
        public void setProductName(String productName) { this.productName = productName; }
        public Long getSalesCount() { return salesCount; }
        public void setSalesCount(Long salesCount) { this.salesCount = salesCount; }
        public Long getSalesAmount() { return salesAmount; }
        public void setSalesAmount(Long salesAmount) { this.salesAmount = salesAmount; }
    }

    /**
     * Sales by hour info DTO
     */
    public static class SalesByHourInfo {
        private Integer hour;
        private Long sales;
        private Integer orders;

        public Integer getHour() { return hour; }
        public void setHour(Integer hour) { this.hour = hour; }
        public Long getSales() { return sales; }
        public void setSales(Long sales) { this.sales = sales; }
        public Integer getOrders() { return orders; }
        public void setOrders(Integer orders) { this.orders = orders; }
    }
}