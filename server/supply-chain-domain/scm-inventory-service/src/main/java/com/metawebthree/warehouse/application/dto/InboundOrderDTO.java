package com.metawebthree.warehouse.application.dto;

import lombok.Data;
import java.time.LocalDateTime;
import java.util.List;

@Data
public class InboundOrderDTO {
    private Long id;
    private String orderNo;
    private String inboundType;
    private Long warehouseId;
    private String warehouseName;
    private String supplierCode;
    private String status;
    private String remark;
    private String operator;
    private LocalDateTime planArrivalTime;
    private LocalDateTime actualArrivalTime;
    private LocalDateTime completedAt;
    private LocalDateTime createdAt;
    private List<InboundOrderItemDTO> items;
}