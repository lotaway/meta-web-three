package com.metawebthree.mes.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.metawebthree.mes.domain.entity.AndonEvent;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("mes_andon_event")
public class AndonEventDO {
    @TableId(type = IdType.ASSIGN_ID)
    private Long id;
    private String eventNo;
    private Long andonTypeId;
    private String andonTypeCode;
    private String andonTypeName;
    private Long andonLevelId;
    private String levelCode;
    private String levelName;
    private String triggerMethod;
    private String sourceSystem;
    private String sourceId;
    private String workshopId;
    private Long workstationId;
    private String equipmentId;
    private String productCode;
    private String workOrderNo;
    private String description;
    private String photoUrl;
    private String reporterId;
    private String reporterName;
    private AndonEvent.AndonEventStatus status;
    private String currentHandlerId;
    private String currentHandlerName;
    private LocalDateTime occurredAt;
    private LocalDateTime acknowledgedAt;
    private LocalDateTime resolvedAt;
    private Integer escalationCount;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}