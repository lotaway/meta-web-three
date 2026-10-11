package com.metawebthree.production.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("production_schedule")
public class ProductionScheduleDO {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private String scheduleCode;
    private String orderCode;
    private String stationCode;
    private Integer sequence;
    private String status;
    private LocalDateTime plannedStartTime;
    private LocalDateTime plannedEndTime;
    private LocalDateTime actualStartTime;
    private LocalDateTime actualEndTime;
    private Integer plannedQuantity;
    private Integer completedQuantity;
    private Double progressPercentage;
    private String processRouteCode;
    private Integer processSequence;
    private String requiredSkills;
    private Integer estimatedDuration;
    private Integer actualDuration;
    private String notes;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}