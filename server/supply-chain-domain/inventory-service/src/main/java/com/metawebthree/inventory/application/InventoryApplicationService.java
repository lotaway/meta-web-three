package com.metawebthree.inventory.application;

import com.metawebthree.inventory.application.dto.InventoryDTO;
import com.metawebthree.inventory.application.dto.InventoryOperationResult;
import com.metawebthree.inventory.application.dto.ReserveInventoryDTO;
import java.util.List;

public interface InventoryApplicationService {

    InventoryDTO queryBySku(String skuCode, Long warehouseId);

    List<InventoryDTO> queryBySkuCode(String skuCode);

    InventoryOperationResult reserve(ReserveInventoryDTO dto);

    InventoryOperationResult confirm(String bizId);

    InventoryOperationResult cancel(String bizId);

    InventoryOperationResult increase(String skuCode, Long warehouseId, 
            Integer quantity, String remark);

    InventoryOperationResult decrease(String skuCode, Long warehouseId, 
            Integer quantity, String remark);
}