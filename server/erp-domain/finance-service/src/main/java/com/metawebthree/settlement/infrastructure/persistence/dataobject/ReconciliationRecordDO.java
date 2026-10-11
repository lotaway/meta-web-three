package com.metawebthree.settlement.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
@TableName("reconciliation_record")
public class ReconciliationRecordDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private String recordNo;
    private String type;
    private LocalDateTime reconcileDate;
    private String channel;
    private BigDecimal totalAmount;
    private Integer totalCount;
    private BigDecimal matchedAmount;
    private Integer matchedCount;
    private BigDecimal unmatchedAmount;
    private Integer unmatchedCount;
    private String status;
    private String remark;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Integer version;
}