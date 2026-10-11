package com.metawebthree.mes.infrastructure.persistence.dataobject.labor;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;

@Data
@TableName("mes_attendance")
public class AttendanceDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private Long operatorId;
    private String operatorCode;
    private String operatorName;
    private LocalDate attendanceDate;
    private LocalTime clockIn;
    private LocalTime clockOut;
    private LocalTime scheduledStart;
    private LocalTime scheduledEnd;
    private String status;
    private Boolean overtime;
    private String remark;
    private String createdBy;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}
