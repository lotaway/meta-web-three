package com.metawebthree.supplier.application.dto;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;

@Data
public class SupplierShipmentNoticeDTO {
    private Long id;
    private String noticeNo;
    private String supplierCode;
    private String orderNo;
    private Long warehouseId;
    private LocalDateTime expectedShipmentDate;
    private LocalDateTime actualShipmentDate;
    private String shipmentMethod;
    private String carrierName;
    private String carrierContact;
    private String trackingNumber;
    private String vehicleNumber;
    private String driverName;
    private String driverPhone;
    private BigDecimal totalQuantity;
    private BigDecimal totalWeight;
    private BigDecimal totalVolume;
    private String status;
    private String remark;
    private String confirmer;
    private LocalDateTime confirmedAt;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private List<SupplierShipmentNoticeItemDTO> items;

    @Data
    public static class SupplierShipmentNoticeItemDTO {
        private Long id;
        private String productCode;
        private String productName;
        private String unit;
        private BigDecimal quantity;
        private BigDecimal weight;
        private BigDecimal volume;
        private String batchNo;
        private LocalDateTime productionDate;
        private LocalDateTime expiryDate;
    }
}