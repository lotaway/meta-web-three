package com.metawebthree.inventory.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("inventory_alert_config")
public class InventoryAlertConfigDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private String configCode;
    private String warehouseCode;
    private String skuCode;
    private Integer safetyStockThreshold;
    private String level;
    private Boolean enabled;
    private Integer cooldownMinutes;
    private String notificationChannels;
    private String notifyUsers;
    private String createdBy;
    private LocalDateTime createdAt;
    private String updatedBy;
    private LocalDateTime updatedAt;
    private Integer version;
}