package com.metawebthree.inventory.application;

import com.metawebthree.inventory.application.dto.InventoryDTO;
import com.metawebthree.inventory.application.dto.InventoryOperationResult;
import com.metawebthree.inventory.application.dto.ReserveInventoryDTO;
import com.metawebthree.inventory.application.dto.StockInDTO;
import java.util.List;

public interface InventoryApplicationService {

    InventoryDTO queryBySku(String skuCode, Long warehouseId);

    List<InventoryDTO> queryBySkuCode(String skuCode);

    InventoryOperationResult reserve(ReserveInventoryDTO dto);

    InventoryOperationResult confirm(String bizId);

    InventoryOperationResult cancel(String bizId);

    InventoryOperationResult increase(String skuCode, Long warehouseId, 
            Integer quantity, String remark);

    /**
     * Record accepted return (RMA) goods back into sellable stock.
     *
     * <p>Idempotent: the same RMA line ({@code rmaNo + skuCode}) is only applied
     * once, even when the request is replayed. Only goods that passed inspection
     * reach this method; rejected/scrapped items are never sent here.</p>
     */
    InventoryOperationResult stockIn(StockInDTO dto);

    InventoryOperationResult decrease(String skuCode, Long warehouseId, 
            Integer quantity, String remark);

    /**
     * Convenience method to reserve inventory for order.
     * @param productId product ID
     * @param quantity quantity to reserve
     * @param orderId order ID for bizId
     * @return operation result
     */
    default InventoryOperationResult reserveInventory(Long productId, Integer quantity, String orderId) {
        ReserveInventoryDTO dto = new ReserveInventoryDTO();
        dto.setSkuCode(productId.toString());
        dto.setQuantity(quantity);
        dto.setBizId(orderId);
        dto.setBizType("ORDER");
        return reserve(dto);
    }
}