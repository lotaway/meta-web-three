package com.metawebthree.mes.infrastructure.persistence.dataobject.scheduling;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
@TableName("mes_schedule_order")
public class ScheduleOrderDO {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String scheduleNo;
    private String orderNo;
    private String productCode;
    private String productName;
    private BigDecimal quantity;
    private BigDecimal completedQuantity;
    private LocalDateTime dueDate;
    private LocalDateTime scheduledStartTime;
    private LocalDateTime scheduledEndTime;
    private LocalDateTime actualStartTime;
    private LocalDateTime actualEndTime;
    private String priority;
    private String status;
    private String workshopId;
    private String routeCode;
    private String remark;
    private String createdBy;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
