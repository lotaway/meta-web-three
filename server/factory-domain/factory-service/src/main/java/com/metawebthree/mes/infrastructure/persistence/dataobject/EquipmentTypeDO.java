package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.metawebthree.mes.domain.entity.EquipmentType;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_equipment_type")
public class EquipmentTypeDO {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    private String typeCode;
    private String typeName;
    private String description;
    private String category;
    private Long statusMachineId;
    private String icon;
    private Integer sortOrder;
    private String attributes;
    private Boolean isActive;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}