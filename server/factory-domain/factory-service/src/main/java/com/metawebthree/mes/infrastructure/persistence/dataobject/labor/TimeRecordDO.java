package com.metawebthree.mes.infrastructure.persistence.dataobject.labor;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;

@Data
@TableName("mes_time_record")
public class TimeRecordDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private Long operatorId;
    private String operatorCode;
    private String operatorName;
    private String workOrderNo;
    private String taskNo;
    private String operationCode;
    private String workCenterId;
    private LocalDate recordDate;
    private LocalDateTime startTime;
    private LocalDateTime endTime;
    private BigDecimal totalHours;
    private String recordType;
    private String status;
    private String approvedBy;
    private LocalDateTime approvedAt;
    private String remark;
    private String createdBy;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
