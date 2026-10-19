package com.metawebthree.rma.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@TableName("rma_stock_in")
public class RmaStockInRecordDO {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    private Long rmaId;
    private String rmaNo;
    private String skuCode;
    private String skuName;
    private Long warehouseId;
    private Integer quantity;
    private String status;
    private Integer retryCount;
    private String lastError;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
