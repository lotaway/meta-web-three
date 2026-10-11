package com.metawebthree.supplier.domain.entity;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
public class Supplier {
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
    private String name;
    private String contactPerson;
    private String contactPhone;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public boolean isActive() {
        return "ACTIVE".equals(status);
    }

    public void activate() {
        this.status = "ACTIVE";
    }

    public void deactivate() {
        this.status = "INACTIVE";
    }
}