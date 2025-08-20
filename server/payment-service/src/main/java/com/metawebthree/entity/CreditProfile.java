package com.metawebthree.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

@TableName("Credit_Profile")
@Data
public class CreditProfile {

    @TableId
    private Long userId;

    @TableField("base_credit_limit")
    private int baseCreditLimit = 10000;

    @TableField("current_credit_limit")
    private int currentCreditLimit = 10000;

    @TableField("credit_used")
    private int creditUsed = 0;

    @TableField(value = "risk_level")
    private String riskLevel = "C";

    @TableField("last_score")
    private Integer lastScore;

    @TableField("last_limit_adjustment")
    private LocalDateTime lastLimitAdjustment;

    @TableField("overdue_count_3m")
    private int overdueCount3m = 0;

    @TableField("transaction_success_rate")
    private double transactionSuccessRate = 100.00;

    @TableField("credit_utilization_rate")
    private double creditUtilizationRate = 0.00;

    @TableField("last_score_change")
    private Integer lastScoreChange;

    @TableField("max_adjustment_percentage")
    private double maxAdjustmentPercentage = 0.15;

    @TableField(value = "adjustment_history")
    private List<Map<String, Object>> adjustmentHistory;

    @TableField("last_update")
    private LocalDateTime lastUpdate = LocalDateTime.now();
}
