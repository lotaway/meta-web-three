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
    private Long equipmentTypeId;
    private String equipmentType;
    private String workshopId;
    private String workstationId;
    private String status;
    private Long statusConfigId;
    private BigDecimal utilizationRate;
    private Integer todayOutput;
    private String currentTaskNo;
    private LocalDateTime lastMaintenanceTime;
    private LocalDateTime nextMaintenanceTime;
    private Long totalRunningSeconds;
    private Long totalIdleSeconds;
    private Long totalDowntimeSeconds;
    
    private String digitalTwinDeviceCode;
    private BigDecimal positionX;
    private BigDecimal positionY;
    private BigDecimal positionZ;
    private BigDecimal rotationY;
    private String ipAddress;
    private String macAddress;
    private String mqttTopic;
    private LocalDateTime lastHeartbeat;
    
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}