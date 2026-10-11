package com.metawebthree.inventory.application.dto;

import lombok.Data;
import java.time.LocalDate;
import java.time.LocalDateTime;

@Data
public class DemandForecastDTO {
    private Long id;
    private String skuCode;
    private Long warehouseId;
    private Integer forecastPeriodDays;
    private Integer predictedQuantity;
    private Integer confidenceLevel;
    private String forecastMethod;
    private LocalDate forecastStartDate;
    private LocalDate forecastEndDate;
    private String status;
    private LocalDateTime generatedAt;
    private String notes;
}