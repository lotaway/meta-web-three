package com.metawebthree.inventory.application;

import com.metawebthree.inventory.application.dto.InventoryDTO;
import com.metawebthree.inventory.application.dto.InventoryOperationResult;
import com.metawebthree.inventory.application.dto.ReserveInventoryDTO;
import com.metawebthree.supplychain.generated.rpc.ReleaseInventoryByOrderIdRequest;
import com.metawebthree.supplychain.generated.rpc.ReleaseInventoryByOrderIdResponse;
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

    public ReleaseInventoryByOrderIdResponse releaseInventoryByOrderId(ReleaseInventoryByOrderIdRequest request) {
        try {
            Long orderId = request.getOrderId();
            String reason = request.getReason();
            
            // Cancel reservation by orderId (bizId)
            InventoryOperationResult result = appService.cancel(orderId.toString());
            
            ReleaseInventoryByOrderIdResponse.Builder builder = ReleaseInventoryByOrderIdResponse.newBuilder()
                    .setSuccess(result.isSuccess())
                    .setMessage(result.getMessage());
            
            return builder.build();
        } catch (Exception e) {
            return ReleaseInventoryByOrderIdResponse.newBuilder()
                    .setSuccess(false)
                    .setMessage("释放库存失败: " + e.getMessage())
                    .build();
        }
    }
}