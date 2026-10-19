package com.metawebthree.inventory.application.dto;

import lombok.Data;

/**
 * Request to record accepted return goods back into sellable stock.
 *
 * <p>The pair {@code rmaNo + skuCode} identifies the RMA line and is used to
 * build the idempotency business key, so replaying the same RMA line never
 * increases stock more than once.</p>
 */
@Data
public class StockInDTO {
    private Long rmaId;
    private String rmaNo;
    private String skuCode;
    private String skuName;
    private Long warehouseId;
    private Integer quantity;
}
