package com.metawebthree.rma.application.dto;

import lombok.Data;
import java.math.BigDecimal;

@Data
public class MakeDispositionRequest {
    private String dispositionType;
    private String dispositionBy;
    private BigDecimal refundAmount;
    private String replacementSkuCode;
    private Integer replacementQuantity;
    private Integer scrapQuantity;
    private String scrapReason;
    private String remark;
}
