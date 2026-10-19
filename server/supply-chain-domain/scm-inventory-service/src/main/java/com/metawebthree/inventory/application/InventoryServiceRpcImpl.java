package com.metawebthree.inventory.application;

import com.metawebthree.inventory.application.dto.InventoryDTO;
import com.metawebthree.inventory.application.dto.InventoryOperationResult;
import com.metawebthree.inventory.application.dto.StockInDTO;
import com.metawebthree.supplychain.generated.rpc.*;
import lombok.extern.slf4j.Slf4j;
import org.apache.dubbo.config.annotation.DubboService;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.concurrent.CompletableFuture;

@Slf4j
@DubboService
@Component
public class InventoryServiceRpcImpl implements InventoryService {

    private final InventoryApplicationService appService;

    public InventoryServiceRpcImpl(InventoryApplicationService appService) {
        this.appService = appService;
    }

    @Override
    public GetInventoryResponse getInventory(GetInventoryRequest request) {
        log.info("Dubbo call: getInventory, productId={}, warehouseCode={}", request.getProductId(), request.getWarehouseCode());
        try {
            String skuCode = String.valueOf(request.getProductId());
            List<InventoryDTO> inventories = appService.queryBySkuCode(skuCode);
            if (inventories.isEmpty()) {
                return GetInventoryResponse.newBuilder().build();
            }
            return buildGetInventoryResponse(request, inventories.get(0));
        } catch (Exception e) {
            log.error("Failed to get inventory", e);
            return GetInventoryResponse.newBuilder().build();
        }
    }

    private GetInventoryResponse buildGetInventoryResponse(GetInventoryRequest request, InventoryDTO inv) {
        return GetInventoryResponse.newBuilder()
                .setProductId(request.getProductId())
                .setWarehouseCode(inv.getWarehouseName() != null ? inv.getWarehouseName() : "")
                .setAvailableQty(inv.getAvailableQuantity() != null ? inv.getAvailableQuantity() : 0)
                .setReservedQty(inv.getReservedQuantity() != null ? inv.getReservedQuantity() : 0)
                .setTotalQty(inv.getTotalQuantity() != null ? inv.getTotalQuantity() : 0)
                .build();
    }

    @Override
    public CompletableFuture<GetInventoryResponse> getInventoryAsync(GetInventoryRequest request) {
        return CompletableFuture.completedFuture(getInventory(request));
    }

    @Override
    public ReserveInventoryResponse reserveInventory(ReserveInventoryRequest request) {
        log.info("Dubbo call: reserveInventory, orderId={}, productId={}, quantity={}", request.getOrderId(), request.getProductId(), request.getQuantity());
        try {
            InventoryOperationResult result = appService.reserveInventory(
                    request.getProductId(),
                    request.getQuantity(),
                    String.valueOf(request.getOrderId()));
            return ReserveInventoryResponse.newBuilder()
                    .setSuccess(result.isSuccess())
                    .setMessage(result.getMessage())
                    .setReservationId(result.getBizId() != null ? result.getBizId() : "")
                    .build();
        } catch (Exception e) {
            log.error("Failed to reserve inventory", e);
            return ReserveInventoryResponse.newBuilder()
                    .setSuccess(false)
                    .setMessage("预订库存失败: " + e.getMessage())
                    .build();
        }
    }

    @Override
    public CompletableFuture<ReserveInventoryResponse> reserveInventoryAsync(ReserveInventoryRequest request) {
        return CompletableFuture.completedFuture(reserveInventory(request));
    }

    @Override
    public ReleaseInventoryResponse releaseInventory(ReleaseInventoryRequest request) {
        log.info("Dubbo call: releaseInventory, reservationId={}", request.getReservationId());
        try {
            InventoryOperationResult result = appService.cancel(request.getReservationId());
            return ReleaseInventoryResponse.newBuilder()
                    .setSuccess(result.isSuccess())
                    .setMessage(result.getMessage())
                    .build();
        } catch (Exception e) {
            log.error("Failed to release inventory", e);
            return ReleaseInventoryResponse.newBuilder()
                    .setSuccess(false)
                    .setMessage("释放库存失败: " + e.getMessage())
                    .build();
        }
    }

    @Override
    public CompletableFuture<ReleaseInventoryResponse> releaseInventoryAsync(ReleaseInventoryRequest request) {
        return CompletableFuture.completedFuture(releaseInventory(request));
    }

    @Override
    public ReleaseInventoryByOrderIdResponse releaseInventoryByOrderId(ReleaseInventoryByOrderIdRequest request) {
        log.info("Dubbo call: releaseInventoryByOrderId, orderId={}", request.getOrderId());
        try {
            InventoryOperationResult result = appService.cancel(String.valueOf(request.getOrderId()));
            return ReleaseInventoryByOrderIdResponse.newBuilder()
                    .setSuccess(result.isSuccess())
                    .setMessage(result.getMessage())
                    .build();
        } catch (Exception e) {
            log.error("Failed to release inventory by orderId", e);
            return ReleaseInventoryByOrderIdResponse.newBuilder()
                    .setSuccess(false)
                    .setMessage("释放库存失败: " + e.getMessage())
                    .build();
        }
    }

    @Override
    public CompletableFuture<ReleaseInventoryByOrderIdResponse> releaseInventoryByOrderIdAsync(ReleaseInventoryByOrderIdRequest request) {
        return CompletableFuture.completedFuture(releaseInventoryByOrderId(request));
    }

    @Override
    public ConfirmInventoryResponse confirmInventory(ConfirmInventoryRequest request) {
        log.info("Dubbo call: confirmInventory, orderId={}", request.getOrderId());
        try {
            InventoryOperationResult result = appService.confirm(String.valueOf(request.getOrderId()));
            return ConfirmInventoryResponse.newBuilder()
                    .setSuccess(result.isSuccess())
                    .setMessage(result.getMessage())
                    .build();
        } catch (Exception e) {
            log.error("Failed to confirm inventory by orderId", e);
            return ConfirmInventoryResponse.newBuilder()
                    .setSuccess(false)
                    .setMessage("确认库存失败: " + e.getMessage())
                    .build();
        }
    }

    @Override
    public CompletableFuture<ConfirmInventoryResponse> confirmInventoryAsync(ConfirmInventoryRequest request) {
        return CompletableFuture.completedFuture(confirmInventory(request));
    }

    @Override
    public StockInResponse stockIn(StockInRequest request) {
        log.info("Dubbo call: stockIn, rmaId={}, rmaNo={}, skuCode={}, quantity={}, warehouseId={}",
                request.getRmaId(), request.getRmaNo(), request.getSkuCode(),
                request.getQuantity(), request.getWarehouseId());
        try {
            StockInDTO dto = new StockInDTO();
            dto.setRmaId(request.getRmaId());
            dto.setRmaNo(request.getRmaNo());
            dto.setSkuCode(request.getSkuCode());
            dto.setQuantity(request.getQuantity());
            dto.setWarehouseId(request.getWarehouseId());
            InventoryOperationResult result = appService.stockIn(dto);
            return StockInResponse.newBuilder()
                    .setSuccess(result.isSuccess())
                    .setDuplicated(result.isDuplicated())
                    .setMessage(result.getMessage())
                    .setQuantity(result.getQuantity() != null ? result.getQuantity() : 0)
                    .build();
        } catch (Exception e) {
            log.error("Failed to stock in RMA goods", e);
            return StockInResponse.newBuilder()
                    .setSuccess(false)
                    .setMessage("回库失败: " + e.getMessage())
                    .build();
        }
    }

    @Override
    public CompletableFuture<StockInResponse> stockInAsync(StockInRequest request) {
        return CompletableFuture.completedFuture(stockIn(request));
    }
}
