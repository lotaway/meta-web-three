package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.FieldFill;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

import java.time.LocalDateTime;

/**
 * 工单 DO
 */
@Data
@NoArgsConstructor
@SuperBuilder
@EqualsAndHashCode(callSuper = false)
@TableName("mes_work_order")
public class WorkOrderDO {
    
    private Long id;
    private String workOrderNo;
    private String productCode;
    private String productName;
    private Integer quantity;
    private Integer completedQuantity;
    private String status; // DRAFT, RELEASED, IN_PROGRESS, PAUSED, COMPLETED, CANCELLED
    private String priority; // LOW, NORMAL, HIGH, URGENT
    private String workshopId;
    private String processRouteId;
    private LocalDateTime plannedStartTime;
    private LocalDateTime plannedEndTime;
    private LocalDateTime actualStartTime;
    private LocalDateTime actualEndTime;
    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createdAt;
    @TableField(fill = FieldFill.INSERT_UPDATE)
    private LocalDateTime updatedAt;
}