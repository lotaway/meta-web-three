package com.metawebthree.inventory.application;

import com.metawebthree.inventory.application.dto.InventoryDTO;
import com.metawebthree.inventory.application.dto.InventoryOperationResult;
import com.metawebthree.inventory.application.dto.ReserveInventoryDTO;
import com.metawebthree.inventory.domain.entity.Inventory;
import com.metawebthree.inventory.domain.service.InventoryDomainService;
import org.springframework.stereotype.Service;
import java.util.List;

@Service
public class InventoryApplicationServiceImpl implements InventoryApplicationService {

    private final InventoryDomainService domainService;

    public InventoryApplicationServiceImpl(InventoryDomainService domainService) {
        this.domainService = domainService;
    }

    @Override
    public InventoryDTO queryBySku(String skuCode, Long warehouseId) {
        return domainService.findBySkuAndWarehouse(skuCode, warehouseId)
            .map(this::toDTO)
            .orElse(null);
    }

    @Override
    public List<InventoryDTO> queryBySkuCode(String skuCode) {
        return List.of();
    }

    @Override
    public InventoryOperationResult reserve(ReserveInventoryDTO dto) {
        try {
            var inventory = domainService.findBySkuAndWarehouse(
                dto.getSkuCode(), dto.getWarehouseId())
                .orElseGet(() -> domainService.create(dto.getSkuCode(), dto.getWarehouseId()));

            domainService.reserve(inventory, dto.getQuantity(), dto.getBizId());
            return InventoryOperationResult.success(dto.getBizId(), dto.getQuantity());
        } catch (Exception e) {
            return InventoryOperationResult.fail(e.getMessage());
        }
    }

    @Override
    public InventoryOperationResult confirm(String bizId) {
        return InventoryOperationResult.success(bizId, 0);
    }

    @Override
    public InventoryOperationResult cancel(String bizId) {
        return InventoryOperationResult.success(bizId, 0);
    }

    @Override
    public InventoryOperationResult increase(String skuCode, Long warehouseId,
            Integer quantity, String remark) {
        try {
            var inventory = domainService.findBySkuAndWarehouse(skuCode, warehouseId)
                .orElseGet(() -> domainService.create(skuCode, warehouseId));
            domainService.increase(inventory, quantity);
            return InventoryOperationResult.success(null, quantity);
        } catch (Exception e) {
            return InventoryOperationResult.fail(e.getMessage());
        }
    }

    @Override
    public InventoryOperationResult decrease(String skuCode, Long warehouseId,
            Integer quantity, String remark) {
        try {
            var inventory = domainService.findBySkuAndWarehouse(skuCode, warehouseId)
                .orElseThrow(() -> new IllegalStateException("Inventory not found"));
            domainService.decrease(inventory, quantity);
            return InventoryOperationResult.success(null, quantity);
        } catch (Exception e) {
            return InventoryOperationResult.fail(e.getMessage());
        }
    }

    private InventoryDTO toDTO(Inventory inventory) {
        InventoryDTO dto = new InventoryDTO();
        dto.setId(inventory.getId());
        dto.setSkuCode(inventory.getSkuCode());
        dto.setWarehouseId(inventory.getWarehouseId());
        dto.setTotalQuantity(inventory.getTotalQuantity());
        dto.setAvailableQuantity(inventory.getAvailableQuantity());
        dto.setReservedQuantity(inventory.getReservedQuantity());
        dto.setDefectiveQuantity(inventory.getDefectiveQuantity());
        dto.setUnitCost(inventory.getUnitCost());
        dto.setCreatedAt(inventory.getCreatedAt());
        dto.setUpdatedAt(inventory.getUpdatedAt());
        return dto;
    }
}