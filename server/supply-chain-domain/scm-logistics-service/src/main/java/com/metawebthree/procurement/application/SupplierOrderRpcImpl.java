package com.metawebthree.procurement.application;

import java.util.List;
import java.util.concurrent.CompletableFuture;

import org.apache.dubbo.config.annotation.DubboService;

import com.metawebthree.procurement.application.dto.ProcurementOrderDTO;
import com.metawebthree.supplychain.generated.rpc.GetSupplierOrderRequest;
import com.metawebthree.supplychain.generated.rpc.GetSupplierOrderResponse;
import com.metawebthree.supplychain.generated.rpc.ListSupplierOrdersRequest;
import com.metawebthree.supplychain.generated.rpc.ListSupplierOrdersResponse;
import com.metawebthree.supplychain.generated.rpc.SupplierOrderService;
import com.metawebthree.supplychain.generated.rpc.SupplierOrderView;

/**
 * Server-side implementation of the supplier portal order reads. The supplier
 * portal (scm-inventory-service) calls this instead of the old Feign endpoint.
 */
@DubboService
public class SupplierOrderRpcImpl implements SupplierOrderService {

    private final ProcurementApplicationService appService;

    public SupplierOrderRpcImpl(ProcurementApplicationService appService) {
        this.appService = appService;
    }

    @Override
    public ListSupplierOrdersResponse listSupplierOrders(ListSupplierOrdersRequest request) {
        List<ProcurementOrderDTO> orders = appService.listOrders(blankToNull(request.getStatus()));
        ListSupplierOrdersResponse.Builder response = ListSupplierOrdersResponse.newBuilder();
        if (orders != null) {
            orders.stream().map(this::toView).forEach(response::addOrders);
        }
        return response.build();
    }

    @Override
    public CompletableFuture<ListSupplierOrdersResponse> listSupplierOrdersAsync(ListSupplierOrdersRequest request) {
        return CompletableFuture.completedFuture(listSupplierOrders(request));
    }

    @Override
    public GetSupplierOrderResponse getSupplierOrder(GetSupplierOrderRequest request) {
        ProcurementOrderDTO order = appService.queryOrder(request.getOrderNo());
        GetSupplierOrderResponse.Builder response = GetSupplierOrderResponse.newBuilder();
        if (order != null) {
            response.setOrder(toView(order));
        }
        return response.build();
    }

    @Override
    public CompletableFuture<GetSupplierOrderResponse> getSupplierOrderAsync(GetSupplierOrderRequest request) {
        return CompletableFuture.completedFuture(getSupplierOrder(request));
    }

    private SupplierOrderView toView(ProcurementOrderDTO order) {
        SupplierOrderView.Builder view = SupplierOrderView.newBuilder()
                .setOrderNo(nullToEmpty(order.getOrderNo()))
                .setSupplierCode(nullToEmpty(order.getSupplierCode()))
                .setSupplierName(nullToEmpty(order.getSupplierName()))
                .setPurchaseType(nullToEmpty(order.getPurchaseType()))
                .setStatus(nullToEmpty(order.getStatus()))
                .setTotalAmount(order.getTotalAmount() == null ? "" : order.getTotalAmount().toPlainString())
                .setCurrency(nullToEmpty(order.getCurrency()))
                .setPaymentTerms(nullToEmpty(order.getPaymentTerms()))
                .setDeliveryTerms(nullToEmpty(order.getDeliveryTerms()))
                .setRemark(nullToEmpty(order.getRemark()))
                .setExpectedDeliveryDate(toMillis(order.getExpectedDeliveryDate()))
                .setActualDeliveryDate(toMillis(order.getActualDeliveryDate()))
                .setCreatedAt(toMillis(order.getCreatedAt()));
        return view.build();
    }

    private static long toMillis(java.time.LocalDateTime value) {
        return value == null ? 0L : value.atZone(java.time.ZoneId.systemDefault()).toInstant().toEpochMilli();
    }

    private static String blankToNull(String value) {
        return value == null || value.isEmpty() ? null : value;
    }

    private static String nullToEmpty(String value) {
        return value == null ? "" : value;
    }
}
