package com.metaweb.datasource.pipeline.service;

import com.metaweb.datasource.pipeline.model.OrderEvent;
import com.metaweb.datasource.pipeline.model.InventoryEvent;
import com.metaweb.datasource.pipeline.model.UserBehaviorEvent;
import com.metaweb.datasource.pipeline.repository.ClickHouseRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

@Slf4j
@Service
public class EtlService {

    @Autowired
    private ClickHouseRepository clickHouseRepository;

    private static final DateTimeFormatter YEAR_MONTH_FMT = DateTimeFormatter.ofPattern("yyyyMM");

    public void processOrderEvent(OrderEvent event) {
        try {
            OrderAnalytics analytics = transformOrderEvent(event);
            clickHouseRepository.insertOrderAnalytics(analytics);
        } catch (Exception e) {
            log.error("Failed to process order event: {}", event.getOrderId(), e);
            throw new RuntimeException("ETL processing failed for order event", e);
        }
    }

    public void processInventoryEvent(InventoryEvent event) {
        try {
            InventoryAnalytics analytics = transformInventoryEvent(event);
            clickHouseRepository.insertInventoryAnalytics(analytics);
        } catch (Exception e) {
            log.error("Failed to process inventory event: {}", event.getProductId(), e);
            throw new RuntimeException("ETL processing failed for inventory event", e);
        }
    }

    public void processUserBehaviorEvent(UserBehaviorEvent event) {
        try {
            UserBehaviorAnalytics analytics = transformUserBehaviorEvent(event);
            clickHouseRepository.insertUserBehaviorAnalytics(analytics);
        } catch (Exception e) {
            log.error("Failed to process user behavior event: {}", event.getEventId(), e);
            throw new RuntimeException("ETL processing failed for user behavior event", e);
        }
    }

    private OrderAnalytics transformOrderEvent(OrderEvent event) {
        OrderAnalytics analytics = new OrderAnalytics();
        analytics.setEventId(event.getEventId());
        analytics.setEventType(event.getEventType());
        analytics.setOrderId(event.getOrderId());
        analytics.setUserId(event.getUserId());
        analytics.setTotalAmount(event.getTotalAmount());
        analytics.setStatus(event.getStatus());
        analytics.setEventTime(event.getEventTime());
        analytics.setProductInfo(event.getProductInfo());
        analytics.setPaymentMethod(event.getPaymentMethod());
        analytics.setMerchantId(event.getMerchantId());
        analytics.setProcessedTime(LocalDateTime.now());
        analytics.setYearMonth(event.getEventTime().format(YEAR_MONTH_FMT));
        analytics.setDayOfWeek(event.getEventTime().getDayOfWeek().getValue());
        analytics.setHourOfDay(event.getEventTime().getHour());
        return analytics;
    }

    private InventoryAnalytics transformInventoryEvent(InventoryEvent event) {
        InventoryAnalytics analytics = new InventoryAnalytics();
        analytics.setEventId(event.getEventId());
        analytics.setEventType(event.getEventType());
        analytics.setProductId(event.getProductId());
        analytics.setProductName(event.getProductName());
        analytics.setQuantity(event.getQuantity());
        analytics.setAvailableQty(event.getAvailableQty());
        analytics.setReservedQty(event.getReservedQty());
        analytics.setWarehouseId(event.getWarehouseId());
        analytics.setEventTime(event.getEventTime());
        analytics.setOperator(event.getOperator());
        analytics.setRemark(event.getRemark());
        analytics.setProcessedTime(LocalDateTime.now());
        return analytics;
    }

    private UserBehaviorAnalytics transformUserBehaviorEvent(UserBehaviorEvent event) {
        UserBehaviorAnalytics analytics = new UserBehaviorAnalytics();
        analytics.setEventId(event.getEventId());
        analytics.setEventType(event.getEventType());
        analytics.setUserId(event.getUserId());
        analytics.setSessionId(event.getSessionId());
        analytics.setPageUrl(event.getPageUrl());
        analytics.setReferrer(event.getReferrer());
        analytics.setProductId(event.getProductId());
        analytics.setSearchKeyword(event.getSearchKeyword());
        analytics.setCategory(event.getCategory());
        analytics.setDuration(event.getDuration());
        analytics.setDeviceType(event.getDeviceType());
        analytics.setBrowser(event.getBrowser());
        analytics.setOs(event.getOs());
        analytics.setIpAddress(event.getIpAddress());
        analytics.setEventTime(event.getEventTime());
        analytics.setExtraData(event.getExtraData());
        analytics.setProcessedTime(LocalDateTime.now());
        if (event.getBrowser() != null) {
            analytics.setBrowserFamily(parseBrowserFamily(event.getBrowser()));
        }
        return analytics;
    }

    private String parseBrowserFamily(String browser) {
        if (browser == null) return "Unknown";
        browser = browser.toLowerCase();
        if (browser.contains("chrome")) return "Chrome";
        if (browser.contains("firefox")) return "Firefox";
        if (browser.contains("safari")) return "Safari";
        if (browser.contains("edge")) return "Edge";
        return "Other";
    }

    @lombok.Data
    public static class OrderAnalytics {
        private String eventId;
        private String eventType;
        private Long orderId;
        private Long userId;
        private java.math.BigDecimal totalAmount;
        private String status;
        private LocalDateTime eventTime;
        private String productInfo;
        private String paymentMethod;
        private Long merchantId;
        private LocalDateTime processedTime;
        private String yearMonth;
        private Integer dayOfWeek;
        private Integer hourOfDay;
    }

    @lombok.Data
    public static class InventoryAnalytics {
        private String eventId;
        private String eventType;
        private Long productId;
        private String productName;
        private Integer quantity;
        private Integer availableQty;
        private Integer reservedQty;
        private String warehouseId;
        private LocalDateTime eventTime;
        private String operator;
        private String remark;
        private LocalDateTime processedTime;
    }

    @lombok.Data
    public static class UserBehaviorAnalytics {
        private String eventId;
        private String eventType;
        private Long userId;
        private String sessionId;
        private String pageUrl;
        private String referrer;
        private Long productId;
        private String searchKeyword;
        private String category;
        private Integer duration;
        private String deviceType;
        private String browser;
        private String os;
        private String ipAddress;
        private LocalDateTime eventTime;
        private String extraData;
        private LocalDateTime processedTime;
        private String browserFamily;
    }
}
