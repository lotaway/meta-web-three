package com.metawebthree.crm.domain.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;

import java.time.LocalDate;
import java.time.LocalDateTime;

@Data
@TableName("crm_contact")
public class Contact {

    @TableId(type = IdType.ASSIGN_ID)
    private Long id;

    private String firstName;

    private String lastName;

    private String email;

    private String phone;

    private String mobile;

    private String position;

    private String department;

    private Long customerId;

    private Boolean isPrimary;

    private String address;

    private String city;

    private String province;

    private String country;

    private String postalCode;

    private LocalDate birthday;

    private String notes;

    private String createdBy;

    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createdAt;

    @TableField(fill = FieldFill.INSERT_UPDATE)
    private LocalDateTime updatedAt;

    @TableLogic
    private Integer deleted;
}
