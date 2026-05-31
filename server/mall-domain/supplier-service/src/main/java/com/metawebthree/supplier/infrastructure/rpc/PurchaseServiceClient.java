package com.metawebthree.supplier.infrastructure.rpc;

import com.metawebthree.supplychain.generated.rpc.ProcurementService;
import com.metawebthree.supplychain.generated.rpc.CreatePurchaseOrderRequest;
import com.metawebthree.supplychain.generated.rpc.CreatePurchaseOrderResponse;
import com.metawebthree.supplychain.generated.rpc.GetPurchaseOrderRequest;
import com.metawebthree.supplychain.generated.rpc.GetPurchaseOrderResponse;
import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.stereotype.Component;

@Component
public class PurchaseServiceClient {

    @DubboReference
    private ProcurementService procurementService;

    public GetPurchaseOrderResponse getPurchaseOrder(String purchaseOrderNo) {
        GetPurchaseOrderRequest request = GetPurchaseOrderRequest.newBuilder()
                .setPurchaseOrderNo(purchaseOrderNo)
                .build();
        return procurementService.getPurchaseOrder(request);
    }

    public CreatePurchaseOrderResponse createPurchaseOrder(String supplierCode) {
        CreatePurchaseOrderRequest request = CreatePurchaseOrderRequest.newBuilder()
                .setSupplierCode(supplierCode)
                .build();
        return procurementService.createPurchaseOrder(request);
    }
}