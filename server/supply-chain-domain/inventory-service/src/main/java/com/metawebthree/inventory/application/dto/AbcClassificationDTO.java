package com.metawebthree.inventory.application.dto;

import lombok.Data;
import java.math.BigDecimal;

@Data
public class AbcClassificationDTO {
    private String skuCode;
    private String category;
    private BigDecimal totalValue;
    private BigDecimal turnoverRate;
    private Integer rank;
}