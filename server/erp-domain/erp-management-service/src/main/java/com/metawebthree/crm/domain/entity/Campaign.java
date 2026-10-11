package com.metawebthree.crm.domain.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;

@Data
@TableName("crm_campaign")
public class Campaign {

    @TableId(type = IdType.ASSIGN_ID)
    private Long id;

    private String name;

    private String description;

    private String type;

    private String status;

    private LocalDate startDate;

    private LocalDate endDate;

    private BigDecimal budget;

    private BigDecimal actualCost;

    private BigDecimal expectedRevenue;

    private String targetAudience;

    private Integer leadsGenerated;

    private Integer convertedCustomers;

    private BigDecimal roi;

    private String createdBy;

    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createdAt;

    @TableField(fill = FieldFill.INSERT_UPDATE)
    private LocalDateTime updatedAt;

    @TableLogic
    private Integer deleted;
}
