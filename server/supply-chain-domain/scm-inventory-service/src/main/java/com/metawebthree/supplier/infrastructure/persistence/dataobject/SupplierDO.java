package com.metawebthree.supplier.infrastructure.persistence.dataobject;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
@TableName("supplier")
public class SupplierDO {
    @TableId(type = IdType.AUTO)
    private Long id;
    private String supplierCode;
    private String supplierName;
    private String supplierType;
    private String businessLicense;
    private String taxId;
    private String province;
    private String city;
    private String district;
    private String address;
    private String contact;
    private String phone;
    private String email;
    private String status;
    private BigDecimal creditLimit;
    private String paymentTerms;
    private String category;
    private Integer score;
    private String level;
    private String assessmentLevel;
    private String contactPerson;
    private String contactPhone;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private Integer version;
}