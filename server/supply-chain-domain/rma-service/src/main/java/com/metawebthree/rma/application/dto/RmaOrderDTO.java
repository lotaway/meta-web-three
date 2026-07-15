package com.metawebthree.rma.application.dto;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;

@Data
public class RmaOrderDTO {
    private Long id;
    private String rmaNo;
    private String orderNo;
    private String returnType;
    private String status;
    private Long customerId;
    private String customerName;
    private String contactPhone;
    private String reasonCode;
    private String reasonDescription;
    private Long warehouseId;
    private Integer totalQuantity;
    private BigDecimal totalAmount;
    private String currency;
    private String createdBy;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private List<RmaOrderItemDTO> items;
    private RmaInspectionDTO inspection;
    private RmaDispositionDTO disposition;
    private ReturnShippingDTO shipping;
}
