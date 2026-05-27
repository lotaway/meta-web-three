package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_equipment_maintenance_plan")
public class EquipmentMaintenancePlanDO {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private String planCode;
    private String planName;
    private String description;
    private Long equipmentTypeId;
    private String equipmentTypeCode;
    private String cycleType;          // TIME_BASED 或 RUNNING_HOURS
    private Integer cycleDays;         // 周期天数
    private Integer cycleRunningHours; // 周期运行时长
    private Integer advanceAlertDays;  // 提前预警天数
    private Boolean isActive;
    
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}