package com.metawebthree.crm.domain.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;

@Data
@TableName("crm_opportunity")
public class Opportunity {

    @TableId(type = IdType.ASSIGN_ID)
    private Long id;

    private String opportunityNo;

    private String title;

    private Long leadId;

    private Long customerId;

    private Long contactId;

    private Long pipelineId;

    private String stage;

    private BigDecimal amount;

    private Integer probability;

    private LocalDate expectedCloseDate;

    private LocalDate actualCloseDate;

    private String competitor;

    private String description;

    private String assignedTo;

    private String createdBy;

    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createdAt;

    @TableField(fill = FieldFill.INSERT_UPDATE)
    private LocalDateTime updatedAt;

    @TableLogic
    private Integer deleted;
}
