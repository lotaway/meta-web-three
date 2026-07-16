package com.metawebthree.dom.infrastructure.rpc;

import com.metawebthree.dom.domain.service.WarehouseInfo;
import com.metawebthree.dom.domain.service.WarehouseServiceClient;
import org.springframework.stereotype.Service;

@Service
public class WarehouseServiceRpcClient implements WarehouseServiceClient {

    // TODO: Replace with actual Dubbo/gRPC stub when proto services are available.
    // For example:
    //   @DubboReference
    //   private WarehouseServiceGrpc.WarehouseServiceBlockingStub warehouseStub;

    @Override
    public WarehouseInfo getWarehouse(Long warehouseId) {
        // TODO: Implement real RPC call:
        //   WarehouseRequest request = WarehouseRequest.newBuilder()
        //       .setWarehouseId(warehouseId)
        //       .build();
        //   WarehouseResponse response = warehouseStub.getWarehouse(request);
        //   return new WarehouseInfo(response.getId(), response.getName(),
        //       response.getRegion(), response.getLatitude(), response.getLongitude());
        throw new UnsupportedOperationException(
                "WarehouseService RPC not configured. Replace WarehouseServiceRpcClient with a real Dubbo/gRPC stub.");
    }

    @Override
    public Double getWarehouseDistance(String fromRegion, Long warehouseId) {
        // TODO: Implement real RPC call:
        //   DistanceRequest request = DistanceRequest.newBuilder()
        //       .setFromRegion(fromRegion)
        //       .setWarehouseId(warehouseId)
        //       .build();
        //   DistanceResponse response = warehouseStub.getDistance(request);
        //   return response.getDistance();
        throw new UnsupportedOperationException(
                "WarehouseService RPC not configured. Replace WarehouseServiceRpcClient with a real Dubbo/gRPC stub.");
    }
}
