package com.metawebthree.inventory.application;

import com.metawebthree.inventory.application.dto.InventoryDTO;
import com.metawebthree.inventory.application.dto.InventoryOperationResult;
import com.metawebthree.inventory.application.dto.ReserveInventoryDTO;
import org.springframework.stereotype.Service;

@Service
public class InventoryServiceRpcImpl {

    private final InventoryApplicationService appService;

    public InventoryServiceRpcImpl(InventoryApplicationService appService) {
        this.appService = appService;
    }

    public InventoryDTO queryInventory(String skuCode, Long warehouseId) {
        return appService.queryBySku(skuCode, warehouseId);
    }

    public InventoryOperationResult reserveInventory(ReserveInventoryDTO dto) {
        return appService.reserve(dto);
    }

    public InventoryOperationResult confirmReservation(String bizId) {
        return appService.confirm(bizId);
    }

    public InventoryOperationResult cancelReservation(String bizId) {
        return appService.cancel(bizId);
    }
}