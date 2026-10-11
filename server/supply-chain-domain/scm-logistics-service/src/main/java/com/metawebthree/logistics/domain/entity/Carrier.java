package com.metawebthree.logistics.domain.entity;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Data
public class Carrier {
    private Long id;
    private String carrierCode;
    private String carrierName;
    private String carrierType;
    private String contact;
    private String phone;
    private String website;
    private String status;
    private BigDecimal baseFreight;
    private BigDecimal weightUnitPrice;
    private BigDecimal volumeUnitPrice;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public boolean isActive() {
        return "ACTIVE".equals(status);
    }
}