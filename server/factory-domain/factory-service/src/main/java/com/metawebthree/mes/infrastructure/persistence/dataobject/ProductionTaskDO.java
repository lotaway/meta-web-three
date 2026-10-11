package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_production_task")
public class ProductionTaskDO {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private String taskNo;
    private Long workOrderId;
    private String workOrderNo;
    private Long workstationId;
    private String workstationName;
    private Integer stepNo;
    private String stepCode;
    private String stepName;
    private String assignedTo;
    private String status;
    private Integer plannedQuantity;
    private Integer completedQuantity;
    private Integer qualifiedQuantity;
    private Integer rejectedQuantity;
    private LocalDateTime startTime;
    private LocalDateTime endTime;
    private Integer actualDurationMinutes;
    private String remarks;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}