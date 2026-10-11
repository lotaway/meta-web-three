package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_equipment_maintenance_item")
public class EquipmentMaintenanceItemDO {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private Long planId;
    private String itemCode;
    private String itemName;
    private String description;
    private String checkMethod;
    private String standard;
    private Boolean isRequired;
    private Integer sortOrder;
    
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}