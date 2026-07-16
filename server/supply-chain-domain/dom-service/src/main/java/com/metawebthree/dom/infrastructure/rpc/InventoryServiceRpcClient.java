package com.metawebthree.dom.infrastructure.rpc;

import com.metawebthree.dom.domain.service.InventoryServiceClient;
import org.springframework.stereotype.Service;

@Service
public class InventoryServiceRpcClient implements InventoryServiceClient {

    // TODO: Replace with actual Dubbo/gRPC stub when proto services are available.
    // For example:
    //   @DubboReference
    //   private InventoryServiceGrpc.InventoryServiceBlockingStub inventoryStub;

    @Override
    public Integer checkInventory(String skuCode, Long warehouseId) {
        // TODO: Implement real RPC call:
        //   InventoryRequest request = InventoryRequest.newBuilder()
        //       .setSkuCode(skuCode)
        //       .setWarehouseId(warehouseId)
        //       .build();
        //   InventoryResponse response = inventoryStub.checkInventory(request);
        //   return response.getAvailableQuantity();
        throw new UnsupportedOperationException(
                "InventoryService RPC not configured. Replace InventoryServiceRpcClient with a real Dubbo/gRPC stub.");
    }
}
