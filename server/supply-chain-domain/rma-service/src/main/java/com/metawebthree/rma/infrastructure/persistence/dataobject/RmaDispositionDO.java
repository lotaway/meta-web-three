package com.metawebthree.rma.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
@TableName("rma_disposition")
public class RmaDispositionDO {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    private Long rmaId;
    private String rmaNo;
    private String dispositionType;
    private BigDecimal refundAmount;
    private String replacementSkuCode;
    private Integer replacementQuantity;
    private Integer scrapQuantity;
    private String scrapReason;
    private String dispositionBy;
    private LocalDateTime dispositionDate;
    private String remark;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
