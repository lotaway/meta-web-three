package com.metawebthree.production.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("work_station")
public class WorkStationDO {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private String stationCode;
    private String stationName;
    private String stationType;
    private String workshopCode;
    private String productionLineCode;
    private String status;
    private Integer capacity;
    private Integer currentLoad;
    private Double efficiency;
    private String currentOperator;
    private String currentOrderCode;
    private Double positionX;
    private Double positionY;
    private String ipAddress;
    private LocalDateTime lastMaintenanceTime;
    private LocalDateTime nextMaintenanceTime;
    private String equipmentList;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}