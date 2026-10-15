package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.metawebthree.mes.domain.entity.EquipmentStatusConfig;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_equipment_status_config")
public class EquipmentStatusConfigDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private Long equipmentTypeId;
    private String statusCode;
    private String statusName;
    private String statusCategory;
    private Boolean isInitial;
    private Boolean isFinal;
    private Boolean isAlarm;
    private String color;
    private String icon;
    private String description;
    private Integer sortOrder;
    private Boolean isActive;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}