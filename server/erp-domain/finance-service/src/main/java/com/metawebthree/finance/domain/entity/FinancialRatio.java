package com.metawebthree.finance.domain.entity;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
public class FinancialRatio {
    private Long id;
    private String ratioType;
    private BigDecimal value;
    private String period;
    private LocalDateTime calculatedAt;
    private Integer version;

    public enum RatioType {
        INVENTORY_TURNOVER,
        ACCOUNTS_RECEIVABLE_TURNOVER_DAYS,
        ACCOUNTS_PAYABLE_TURNOVER_DAYS,
        GROSS_MARGIN
    }

    public enum PeriodType {
        DAILY,
        WEEKLY,
        MONTHLY,
        QUARTERLY,
        YEARLY
    }

    public boolean isValid() {
        return ratioType != null && value != null;
    }

    public String getDisplayName() {
        if (ratioType == null) {
            return "";
        }
        return switch (ratioType) {
            case "INVENTORY_TURNOVER" -> "库存周转率";
            case "ACCOUNTS_RECEIVABLE_TURNOVER_DAYS" -> "应收账款周转天数";
            case "ACCOUNTS_PAYABLE_TURNOVER_DAYS" -> "应付账款周转天数";
            case "GROSS_MARGIN" -> "毛利率";
            default -> ratioType;
        };
    }
}