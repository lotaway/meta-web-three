package com.metawebthree.developerportal.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("api_usage_stats")
public class ApiUsageStats {

    @TableId(type = IdType.AUTO)
    private Long id;

    @TableField("developer_id")
    private String developerId;

    @TableField("key_id")
    private String keyId;

    @TableField("api_endpoint")
    private String apiEndpoint;

    @TableField("http_method")
    private String httpMethod;

    @TableField("stat_time")
    private LocalDateTime statTime;

    @TableField("request_count")
    private Long requestCount = 0L;

    @TableField("success_count")
    private Long successCount = 0L;

    @TableField("error_count")
    private Long errorCount = 0L;

    @TableField("avg_response_time_ms")
    private Double avgResponseTimeMs;

    @TableField("data_transferred_bytes")
    private Long dataTransferredBytes = 0L;

    @TableField("billing_amount_cents")
    private Long billingAmountCents = 0L;

    @TableField("created_at")
    private LocalDateTime createdAt;
}
