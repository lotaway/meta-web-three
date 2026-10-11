package com.metawebthree.supplier.application.dto;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;

@Data
public class SupplierPortalOrderDTO {
    private String orderNo;
    private String supplierCode;
    private String purchaseType;
    private String status;
    private BigDecimal totalAmount;
    private String currency;
    private String paymentTerms;
    private String deliveryTerms;
    private LocalDateTime expectedDeliveryDate;
    private LocalDateTime actualDeliveryDate;
    private String remark;
    private LocalDateTime createdAt;
    private List<SupplierPortalOrderItemDTO> items;
}

@Data
class SupplierPortalOrderItemDTO {
    private String productCode;
    private String productName;
    private String unit;
    private BigDecimal quantity;
    private BigDecimal price;
    private BigDecimal amount;
}