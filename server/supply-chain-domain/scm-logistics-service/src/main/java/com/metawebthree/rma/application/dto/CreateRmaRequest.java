package com.metawebthree.rma.application.dto;

import lombok.Data;
import java.math.BigDecimal;
import java.util.List;

@Data
public class CreateRmaRequest {
    private String orderNo;
    private Long customerId;
    private String customerName;
    private String contactPhone;
    private String reasonCode;
    private String reasonDescription;
    private Long warehouseId;
    private String returnType;
    private String createdBy;
    private List<CreateRmaItem> items;

    @Data
    public static class CreateRmaItem {
        private String skuCode;
        private String skuName;
        private Integer expectedQuantity;
        private BigDecimal unitPrice;
        private String reasonCode;
        private String reasonDescription;
    }
}
