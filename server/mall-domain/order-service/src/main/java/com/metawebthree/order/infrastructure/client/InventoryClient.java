package com.metawebthree.order.infrastructure.client;

import com.metawebthree.common.exception.BusinessException;
import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.supplychain.generated.rpc.*;
import lombok.extern.slf4j.Slf4j;
import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
@Slf4j
public class InventoryClient {

    @DubboReference(check = false, lazy = true)
    private InventoryService inventoryService;

    /**
     * Reserve inventory for order
     * @param orderId order ID
     * @param skuIds SKU ID list
     * @param quantities quantity list
     * @param reason reservation reason
     * @return true if success
     */
    public boolean reserveInventory(Long orderId, List<Long> skuIds, List<Integer> quantities, String reason) {
        try {
            if (skuIds == null || skuIds.isEmpty()) {
                log.warn("No SKUs to reserve for order: {}", orderId);
                return true;
            }

            // Use first SKU for now (simplified)
            Long productId = skuIds.get(0);
            Integer quantity = quantities != null && !quantities.isEmpty() ? quantities.get(0) : 1;

            ReserveInventoryRequest request = ReserveInventoryRequest.newBuilder()
                    .setOrderId(orderId)
                    .setProductId(productId)
                    .setWarehouseCode("DEFAULT")
                    .setQuantity(quantity)
                    .build();

            ReserveInventoryResponse response = inventoryService.reserveInventory(request);
            
            if (response.getSuccess()) {
                log.info("Inventory reserved successfully - orderId: {}, reservationId: {}", 
                        orderId, response.getReservationId());
                return true;
            } else {
                log.warn("Inventory reservation failed - orderId: {}, message: {}", 
                        orderId, response.getMessage());
                return false;
            }
        } catch (Exception e) {
            log.error("Inventory reservation exception - orderId: {}, error: {}", orderId, e.getMessage());
            throw new BusinessException(ResponseStatus.SYSTEM_ERROR, "Failed to reserve inventory: " + e.getMessage());
        }
    }

    /**
     * Confirm inventory reservation (deduct from reserved)
     * @param orderId order ID
     * @param reason confirmation reason
     * @return true if success
     */
    public boolean confirmInventoryReservation(Long orderId, String reason) {
        try {
            // Use orderId to confirm the reservation
            ReleaseInventoryByOrderIdRequest request = ReleaseInventoryByOrderIdRequest.newBuilder()
                    .setOrderId(orderId)
                    .setReason(reason)
                    .build();

            ReleaseInventoryByOrderIdResponse response = inventoryService.releaseInventoryByOrderId(request);
            
            if (response.getSuccess()) {
                log.info("Inventory confirmation successful - orderId: {}, releasedCount: {}", 
                        orderId, response.getReleasedCount());
                return true;
            } else {
                log.warn("Inventory confirmation failed - orderId: {}, message: {}", 
                        orderId, response.getMessage());
                return false;
            }
        } catch (Exception e) {
            log.error("Inventory confirmation exception - orderId: {}, error: {}", orderId, e.getMessage());
            throw new BusinessException(ResponseStatus.SYSTEM_ERROR, "Failed to confirm inventory: " + e.getMessage());
        }
    }

    /**
     * Release inventory by order ID
     * @param orderId order ID
     * @param reason release reason
     * @return true if success
     */
    public boolean releaseInventoryByOrderId(Long orderId, String reason) {
        try {
            ReleaseInventoryByOrderIdRequest request = ReleaseInventoryByOrderIdRequest.newBuilder()
                    .setOrderId(orderId)
                    .setReason(reason)
                    .build();

            ReleaseInventoryByOrderIdResponse response = inventoryService.releaseInventoryByOrderId(request);
            
            if (response.getSuccess()) {
                log.info("Inventory released by order ID successfully - orderId: {}, releasedCount: {}", 
                        orderId, response.getReleasedCount());
                return true;
            } else {
                log.warn("Inventory release by order ID failed - orderId: {}, message: {}", 
                        orderId, response.getMessage());
                return false;
            }
        } catch (Exception e) {
            log.error("Inventory release by order ID exception - orderId: {}, error: {}", orderId, e.getMessage());
            throw new BusinessException(ResponseStatus.SYSTEM_ERROR, "Failed to release inventory: " + e.getMessage());
        }
    }
}