package com.metawebthree.crm.domain.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@TableName("crm_cs_ticket")
public class CustomerServiceTicket {

    @TableId(type = IdType.ASSIGN_ID)
    private Long id;

    private String ticketNo;

    private String title;

    private Long customerId;

    private Long contactId;

    private Long orderId;

    private String type;

    private String priority;

    private String status;

    private String assignedTo;

    private String description;

    private String resolution;

    private String createdBy;

    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createdAt;

    @TableField(fill = FieldFill.INSERT_UPDATE)
    private LocalDateTime updatedAt;

    @TableLogic
    private Integer deleted;
}
