package com.metawebthree.riskcontrol.infrastructure.rpc;

import com.metawebthree.common.exception.BusinessException;
import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.common.generated.rpc.*;
import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
public class OrderClient {

    @DubboReference
    private OrderService orderService;

    public OrderDTO getOrderById(Long orderId) {
        GetOrderByUserIdRequest request = GetOrderByUserIdRequest.newBuilder()
                .setUserId(orderId)
                .build();
        
        try {
            GetOrderByUserIdResponse response = orderService.getOrderByUserId(request);
            if (response.getOrdersList().isEmpty()) {
                return null;
            }
            return response.getOrdersList().get(0);
        } catch (Exception e) {
            throw new BusinessException(ResponseStatus.SYSTEM_ERROR, "Failed to query order: " + e.getMessage());
        }
    }

    public boolean closeOrder(String orderIds, String note) {
        CloseOrderRequest request = CloseOrderRequest.newBuilder()
                .setIds(orderIds)
                .setNote(note)
                .build();
        
        try {
            CloseOrderResponse response = orderService.closeOrder(request);
            return response.getSuccess();
        } catch (Exception e) {
            throw new BusinessException(ResponseStatus.SYSTEM_ERROR, "Failed to close order: " + e.getMessage());
        }
    }

    public String getLogisticsInfo(Long orderId) {
        QueryLogisticsRequest request = QueryLogisticsRequest.newBuilder()
                .setOrderId(orderId)
                .build();
        
        try {
            QueryLogisticsResponse response = orderService.queryLogistics(request);
            return response.getLogisticsCompany() + ":" + response.getTrackingNumber();
        } catch (Exception e) {
            throw new BusinessException(ResponseStatus.SYSTEM_ERROR, "Failed to query logistics: " + e.getMessage());
        }
    }
}