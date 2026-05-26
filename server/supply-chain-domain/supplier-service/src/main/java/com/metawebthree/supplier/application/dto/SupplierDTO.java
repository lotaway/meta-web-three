package com.metawebthree.supplier.application.dto;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
public class SupplierDTO {
    private Long id;
    private String supplierCode;
    private String supplierName;
    private String name;
    private String supplierType;
    private String province;
    private String city;
    private String address;
    private String contact;
    private String contactPerson;
    private String contactPhone;
    private String phone;
    private String email;
    private String status;
    private BigDecimal creditLimit;
    private String paymentTerms;
    private String category;
    private Integer score;
    private String level;
    private String assessmentLevel;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}