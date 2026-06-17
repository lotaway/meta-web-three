package com.metaweb.datasource.pipeline.repository.entity;

import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@TableName("user_behavior_analytics")
public class UserBehaviorAnalytics {
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
