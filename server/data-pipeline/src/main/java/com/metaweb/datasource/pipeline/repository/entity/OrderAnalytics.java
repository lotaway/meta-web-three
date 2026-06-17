package com.metaweb.datasource.pipeline.repository.entity;

import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
@TableName("order_analytics")
public class OrderAnalytics {
    private String eventId;
    private String eventType;
    private Long orderId;
    private Long userId;
    private BigDecimal totalAmount;
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
