package com.metawebthree.project.domain.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.List;

@Data
@TableName("pm_project")
public class Project {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String projectCode;

    private String projectName;

    private String description;

    private String status;

    private Long departmentId;

    private String departmentName;

    private Long managerId;

    private String managerName;

    private LocalDate startDate;

    private LocalDate endDate;

    private BigDecimal budgetAmount;

    private BigDecimal usedAmount;

    private String currency;

    private Integer progress;

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