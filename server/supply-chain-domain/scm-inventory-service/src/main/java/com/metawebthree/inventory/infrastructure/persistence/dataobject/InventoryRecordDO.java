package com.metawebthree.inventory.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("inventory_record")
public class InventoryRecordDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private String skuCode;
    private Long warehouseId;
    private String bizType;
    private String bizId;
    private Integer quantity;
    private Integer beforeQuantity;
    private Integer afterQuantity;
    private String remark;
    private String operator;
    private LocalDateTime createdAt;
}