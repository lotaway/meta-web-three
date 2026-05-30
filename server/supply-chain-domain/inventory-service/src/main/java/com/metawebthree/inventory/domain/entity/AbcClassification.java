package com.metawebthree.inventory.domain.entity;

import lombok.Data;
import java.math.BigDecimal;

@Data
public class AbcClassification {
    private String skuCode;
    private AbcCategory category;
    private BigDecimal totalValue;
    private BigDecimal turnoverRate;
    private Integer rank;

    public enum AbcCategory {
        A,
        B,
        C
    }
}