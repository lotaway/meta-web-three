package com.metaweb.datasource.pipeline.model;

import lombok.Data;
import java.time.LocalDateTime;

@Data
public class UserBehaviorEvent {
    private String eventId;
    private String eventType; // PAGE_VIEW, PRODUCT_VIEW, ADD_TO_CART, SEARCH, CLICK, PURCHASE
    private Long userId;
    private String sessionId;
    private String pageUrl;
    private String referrer;
    private Long productId;
    private String searchKeyword;
    private String category;
    private Integer duration; // seconds
    private String deviceType; // MOBILE, DESKTOP, TABLET
    private String browser;
    private String os;
    private String ipAddress;
    private LocalDateTime eventTime;
    private String extraData; // JSON string for additional data
    
    public UserBehaviorEvent() {
    }
    
    public UserBehaviorEvent(String eventId, String eventType, Long userId, String sessionId) {
        this.eventId = eventId;
        this.eventType = eventType;
        this.userId = userId;
        this.sessionId = sessionId;
        this.eventTime = LocalDateTime.now();
    }
}
