package com.metawebthree.payment.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.*;
import com.metawebthree.common.DO.BaseDO;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;
import lombok.AllArgsConstructor;

import java.math.BigDecimal;
import java.time.LocalDate;

/**
 * Reconciliation difference record - 对账差异记录
 * Records discrepancies found during reconciliation (长款、短款、金额不一致)
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
@SuperBuilder
@TableName("payment_reconciliation_diff")
public class ReconciliationDiffDO extends BaseDO {

    @TableId(type = IdType.ASSIGN_ID)
    private Long id;

    @TableField("reconciliation_date")
    private LocalDate reconciliationDate;

    @TableField("diff_type")
    private DiffType diffType; // MISSING_ORDER(长款), EXTRA_ORDER(短款), AMOUNT_MISMATCH(金额不一致)

    @TableField("order_no")
    private String orderNo;

    @TableField("internal_amount")
    private BigDecimal internalAmount;

    @TableField("external_amount")
    private BigDecimal externalAmount;

    @TableField("amount_difference")
    private BigDecimal amountDifference;

    @TableField("status")
    private DiffStatus status; // PENDING, HANDLED, IGNORED

    @TableField("handle_remark")
    private String handleRemark;

    @TableField("handled_at")
    private java.time.LocalDateTime handledAt;

    @TableField("handled_by")
    private String handledBy;

    public enum DiffType {
        MISSING_ORDER,   // 长款：外部有但内部没有
        EXTRA_ORDER,     // 短款：内部有但外部没有
        AMOUNT_MISMATCH  // 金额不一致
    }

    public enum DiffStatus {
        PENDING,   // 待处理
        HANDLED,   // 已处理
        IGNORED    // 已忽略
    }
}