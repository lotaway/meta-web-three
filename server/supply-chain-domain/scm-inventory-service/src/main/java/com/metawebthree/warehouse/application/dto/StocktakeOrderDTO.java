package com.metawebthree.warehouse.application.dto;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;

@Data
public class StocktakeOrderDTO {
    private Long id;
    private String orderNo;
    private String type;
    private Long warehouseId;
    private String warehouseName;
    private Long locationId;
    private String locationName;
    private String status;
    private String operator;
    private LocalDateTime plannedDate;
    private LocalDateTime startDate;
    private LocalDateTime endDate;
    private Integer totalSkuCount;
    private Integer checkedSkuCount;
    private Integer discrepancyCount;
    private BigDecimal totalDiscrepancyAmount;
    private String remark;
    private String createdBy;
    private LocalDateTime createdAt;
    private List<StocktakeOrderItemDTO> items;
}