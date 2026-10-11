package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_equipment_checklist_template")
public class EquipmentChecklistTemplateDO {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private String templateCode;
    private String templateName;
    private String equipmentTypeCode;
    private String checkPeriodType;
    private Integer checkPeriodValue;
    private String checkPeriodUnit;
    private Long runningHoursThreshold;
    private String alertBeforeHours;
    private String status;
    private Integer version;
    private String remark;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}