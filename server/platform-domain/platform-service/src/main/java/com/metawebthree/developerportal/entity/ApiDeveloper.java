package com.metawebthree.developerportal.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("api_developer")
public class ApiDeveloper {

    @TableId(type = IdType.AUTO)
    private Long id;

    @TableField("developer_id")
    private String developerId;

    @TableField("email")
    private String email;

    @TableField("name")
    private String name;

    @TableField("phone")
    private String phone;

    @TableField("description")
    private String description;

    @TableField("status")
    private DeveloperStatus status = DeveloperStatus.PENDING;

    @TableField("review_note")
    private String reviewNote;

    @TableField("reviewed_by")
    private String reviewedBy;

    @TableField("reviewed_at")
    private LocalDateTime reviewedAt;

    @TableField("daily_quota")
    private Integer dailyQuota = 10000;

    @TableField("monthly_quota")
    private Integer monthlyQuota = 100000;

    @TableField("billing_plan")
    private BillingPlan billingPlan = BillingPlan.FREE;

    @TableField("balance")
    private Long balance = 0L;

    @TableField("created_at")
    private LocalDateTime createdAt;

    @TableField("updated_at")
    private LocalDateTime updatedAt;

    public enum DeveloperStatus {
        PENDING, APPROVED, SUSPENDED, REJECTED
    }

    public enum BillingPlan {
        FREE(0, 10000, 100000),
        BASIC(2900, 50000, 500000),
        PROFESSIONAL(9900, 200000, 2000000),
        ENTERPRISE(0, Integer.MAX_VALUE, Integer.MAX_VALUE);

        private final int monthlyFeeCents;
        private final int dailyQuota;
        private final int monthlyQuota;

        BillingPlan(int monthlyFeeCents, int dailyQuota, int monthlyQuota) {
            this.monthlyFeeCents = monthlyFeeCents;
            this.dailyQuota = dailyQuota;
            this.monthlyQuota = monthlyQuota;
        }

        public int getMonthlyFeeCents() { return monthlyFeeCents; }
        public int getDailyQuota() { return dailyQuota; }
        public int getMonthlyQuota() { return monthlyQuota; }
    }
}
