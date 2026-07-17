package com.metawebthree.developerportal.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("api_subscription")
public class ApiSubscription {

    @TableId(type = IdType.AUTO)
    private Long id;

    @TableField("subscription_id")
    private String subscriptionId;

    @TableField("developer_id")
    private String developerId;

    @TableField("api_pattern")
    private String apiPattern;

    @TableField("status")
    private SubscriptionStatus status = SubscriptionStatus.PENDING;

    @TableField("review_note")
    private String reviewNote;

    @TableField("reviewed_by")
    private String reviewedBy;

    @TableField("reviewed_at")
    private LocalDateTime reviewedAt;

    @TableField("started_at")
    private LocalDateTime startedAt;

    @TableField("ended_at")
    private LocalDateTime endedAt;

    @TableField("reason")
    private String reason;

    @TableField("created_at")
    private LocalDateTime createdAt;

    @TableField("updated_at")
    private LocalDateTime updatedAt;

    public enum SubscriptionStatus {
        PENDING, APPROVED, ACTIVE, SUSPENDED, CANCELLED
    }
}
