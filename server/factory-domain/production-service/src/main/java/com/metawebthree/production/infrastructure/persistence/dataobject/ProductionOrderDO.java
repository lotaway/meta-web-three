package com.metawebthree.production.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("production_order")
public class ProductionOrderDO {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private String orderCode;
    private String productCode;
    private String productName;
    private Integer quantityPlanned;
    private Integer quantityCompleted;
    private String status;
    private String priority;
    private String workshopCode;
    private String productionLineCode;
    private LocalDateTime plannedStartTime;
    private LocalDateTime plannedEndTime;
    private LocalDateTime actualStartTime;
    private LocalDateTime actualEndTime;
    private Double progressPercentage;
    private String orderType;
    private String customerName;
    private String notes;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}