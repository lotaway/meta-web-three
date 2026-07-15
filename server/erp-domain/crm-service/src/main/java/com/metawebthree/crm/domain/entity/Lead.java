package com.metawebthree.crm.domain.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@TableName("crm_lead")
public class Lead {

    @TableId(type = IdType.ASSIGN_ID)
    private Long id;

    private String leadNo;

    private String name;

    private String company;

    private String title;

    private String email;

    private String phone;

    private String mobile;

    private String source;

    private String status;

    private Integer score;

    private String industry;

    private String city;

    private String province;

    private String country;

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
