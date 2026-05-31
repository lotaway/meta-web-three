package com.metawebthree.project.domain.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;

@Data
@TableName("pm_cost_record")
public class CostRecord {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long projectId;

    private String projectName;

    private String costType;

    private String costCode;

    private String costName;

    private BigDecimal amount;

    private String currency;

    private LocalDate costDate;

    private String description;

    private String status;

    private Long departmentId;

    private String departmentName;

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