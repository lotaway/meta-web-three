package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
@TableName("mes_equipment")
public class EquipmentDO {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private String equipmentCode;
    private String equipmentName;
    private String equipmentType;
    private String workshopId;
    private String workstationId;
    private String status;
    private BigDecimal utilizationRate;
    private Integer todayOutput;
    private String currentTaskNo;
    private LocalDateTime lastMaintenanceTime;
    private LocalDateTime nextMaintenanceTime;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}