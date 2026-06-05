package com.metawebthree.mes.infrastructure.persistence.dataobject.scheduling;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@TableName("mes_schedule_resource")
public class ScheduleResourceDO {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String resourceCode;
    private String resourceName;
    private String resourceType;
    private String status;
    private String workshopId;
    private Double capacityPerShift;
    private String calendarCode;
    private String description;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
