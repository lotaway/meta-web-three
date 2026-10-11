package com.metawebthree.inventory.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("outbound_strategy")
public class OutboundStrategyDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private String strategyCode;
    private String strategyName;
    private String strategyType;
    private Long warehouseId;
    private String warehouseCode;
    private String skuCode;
    private String skuCodePattern;
    private Integer priority;
    private String specificBatchNo;
    private Boolean isActive;
    private String remark;
    private String creator;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Integer version;
}