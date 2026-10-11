package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_workstation")
public class WorkstationDO {
    
    @TableId(type = IdType.AUTO)
    private Long id;
    
    private String workstationCode;
    private String workstationName;
    private String workshopId;
    private String workshopName;
    private String type;
    private String status;
    private String location;
    private Integer capacity;
    private String description;
    private String equipmentIds;
    private String equipmentCodes;
    private String toolIds;
    private String toolNames;
    private String operatorIds;
    private String operatorNames;
    private String extensionFields;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}