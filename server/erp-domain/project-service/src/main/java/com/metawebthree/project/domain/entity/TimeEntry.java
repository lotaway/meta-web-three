package com.metawebthree.project.domain.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;

@Data
@TableName("pm_time_entry")
public class TimeEntry {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long projectId;

    private String projectName;

    private Long taskId;

    private String taskName;

    private Long employeeId;

    private String employeeName;

    private LocalDate workDate;

    private BigDecimal hours;

    private String workType;

    private String description;

    private String status;

    private Long approverId;

    private String approverName;

    private LocalDateTime approvedAt;

    @TableField(fill = FieldFill.INSERT)
    private Long createdBy;

    @TableField(fill = FieldFill.INSERT)
    private String creatorName;

    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createdAt;

    @TableField(fill = FieldFill.INSERT_UPDATE)
    private Long updatedBy;

    @TableField(fill = FieldFill.INSERT_UPDATE)
    private LocalDateTime updatedAt;

    private String remark;
}