package com.metawebthree.mes.infrastructure.persistence.dataobject.scheduling;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
@TableName("mes_schedule_operation")
public class ScheduleOperationDO {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long scheduleOrderId;
    private String operationCode;
    private String operationName;
    private Integer sequenceNo;
    private String resourceCode;
    private String resourceName;
    private BigDecimal setupTimeMinutes;
    private BigDecimal processingTimeMinutes;
    private BigDecimal teardownTimeMinutes;
    private String status;
    private LocalDateTime scheduledStartTime;
    private LocalDateTime scheduledEndTime;
}
