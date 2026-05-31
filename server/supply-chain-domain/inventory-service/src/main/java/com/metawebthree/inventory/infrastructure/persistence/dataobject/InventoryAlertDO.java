package com.metawebthree.inventory.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("inventory_alert")
public class InventoryAlertDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private String alertCode;
    private String warehouseCode;
    private String skuCode;
    private String alertType;
    private String level;
    private String title;
    private String description;
    private Integer currentQuantity;
    private Integer thresholdValue;
    private String status;
    private String solution;
    private String acknowledgedBy;
    private LocalDateTime acknowledgedAt;
    private String resolvedBy;
    private LocalDateTime resolvedAt;
    private LocalDateTime occurredAt;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Integer version;
}