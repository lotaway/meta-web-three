package com.metawebthree.traceability.application.dto;

import lombok.Data;

import java.time.LocalDateTime;

@Data
public class ProductInfoDTO {

    private String productId;

    private String productName;

    private String category;

    private String manufacturer;

    private String productionLocation;

    private LocalDateTime productionDate;

    private Boolean isActive;
}