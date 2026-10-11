package com.metawebthree.supplier.infrastructure.persistence.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("supplier")
public class SupplierEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private String supplierCode;

    private String supplierName;

    private String contactPerson;

    private String contactPhone;

    private String contactEmail;

    private String address;

    private Integer status;

    private Integer verificationStatus;

    private String businessLicense;

    private String legalPerson;

    private Integer supplierLevel;

    private Integer score;

    private String remark;

    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createTime;

    @TableField(fill = FieldFill.INSERT_UPDATE)
    private LocalDateTime updateTime;

    @TableLogic
    private Integer deleted;
}