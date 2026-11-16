package com.metawebthree.supplier.infrastructure.rpc;

import java.math.BigDecimal;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.stereotype.Component;

import com.metawebthree.supplier.infrastructure.rpc.dto.ProcurementOrderDTO;
import com.metawebthree.supplychain.generated.rpc.GetSupplierOrderRequest;
import com.metawebthree.supplychain.generated.rpc.GetSupplierOrderResponse;
import com.metawebthree.supplychain.generated.rpc.ListSupplierOrdersRequest;
import com.metawebthree.supplychain.generated.rpc.ListSupplierOrdersResponse;
import com.metawebthree.supplychain.generated.rpc.SupplierOrderService;
import com.metawebthree.supplychain.generated.rpc.SupplierOrderView;

/**
 * Supplier portal order reads. Calls the provider in scm-logistics-service via
 * Dubbo (service-to-service calls bypass the gateway by design).
 *
 * <p>Public method signatures are unchanged from the old Feign client so the
 * application layer does not need to know about the transport.</p>
 */
@Component
public class ProcurementServiceClient {

    @DubboReference(check = false, lazy = true)
    private SupplierOrderService supplierOrderService;

    public List<ProcurementOrderDTO> listOrders(String status) {
        ListSupplierOrdersRequest.Builder request = ListSupplierOrdersRequest.newBuilder();
        if (status != null && !status.isEmpty()) {
            request.setStatus(status);
        }
        ListSupplierOrdersResponse response = supplierOrderService.listSupplierOrders(request.build());
        if (response == null || response.getOrdersCount() == 0) {
            return Collections.emptyList();
        }
        return response.getOrdersList().stream()
                .map(this::toDTO)
                .collect(Collectors.toList());
    }

    public ProcurementOrderDTO queryOrder(String orderNo) {
        if (orderNo == null || orderNo.isEmpty()) {
            return null;
        }
        GetSupplierOrderResponse response = supplierOrderService.getSupplierOrder(
                GetSupplierOrderRequest.newBuilder().setOrderNo(orderNo).build());
        if (response == null || !response.hasOrder()) {
            return null;
        }
        return toDTO(response.getOrder());
    }

    private ProcurementOrderDTO toDTO(SupplierOrderView view) {
        ProcurementOrderDTO dto = new ProcurementOrderDTO();
        dto.setOrderNo(view.getOrderNo());
        dto.setSupplierCode(view.getSupplierCode());
        dto.setSupplierName(view.getSupplierName());
        dto.setPurchaseType(view.getPurchaseType());
        dto.setStatus(view.getStatus());
        dto.setTotalAmount(view.getTotalAmount().isEmpty() ? null : new BigDecimal(view.getTotalAmount()));
        dto.setCurrency(view.getCurrency());
        dto.setPaymentTerms(view.getPaymentTerms());
        dto.setDeliveryTerms(view.getDeliveryTerms());
        dto.setRemark(view.getRemark());
        dto.setExpectedDeliveryDate(toLocalDateTime(view.getExpectedDeliveryDate()));
        dto.setActualDeliveryDate(toLocalDateTime(view.getActualDeliveryDate()));
        dto.setCreatedAt(toLocalDateTime(view.getCreatedAt()));
        return dto;
    }

    private static LocalDateTime toLocalDateTime(long epochMillis) {
        return epochMillis == 0L ? null : LocalDateTime.ofInstant(Instant.ofEpochMilli(epochMillis), ZoneId.systemDefault());
    }
}
