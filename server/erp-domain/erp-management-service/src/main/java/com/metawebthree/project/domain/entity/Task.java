package com.metawebthree.project.domain.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("pm_task")
public class Task {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long projectId;

    private String taskCode;

    private String taskName;

    private String description;

    private String status;

    private Long parentId;

    private Integer level;

    private Integer sort;

    private Long assigneeId;

    private String assigneeName;

    private LocalDateTime plannedStartDate;

    private LocalDateTime plannedEndDate;

    private LocalDateTime actualStartDate;

    private LocalDateTime actualEndDate;

    private Integer progress;

    private Integer estimatedHours;

    private Integer actualHours;

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