package com.metawebthree.inventory.application.dto;

import lombok.Data;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;

@Data
public class BatchAllocationDTO {
    private String skuCode;
    private Long warehouseId;
    private Integer totalRequiredQuantity;
    private Integer totalAllocatedQuantity;
    private String strategyType;
    private List<BatchPickDetail> batches;

    @Data
    public static class BatchPickDetail {
        private Long batchId;
        private String batchNo;
        private Integer allocatedQuantity;
        private LocalDateTime inboundDate;
        private LocalDateTime expiryDate;
        private String locationCode;
        private BigDecimal unitCost;
    }
}