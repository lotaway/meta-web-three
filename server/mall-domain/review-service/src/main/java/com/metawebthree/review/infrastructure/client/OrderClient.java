package com.metawebthree.review.infrastructure.client;

import com.metawebthree.common.exception.BusinessException;
import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.common.generated.rpc.*;
import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.stereotype.Component;

@Component
public class OrderClient {

    @DubboReference
    private OrderService orderService;

    /**
     * Get order by ID
     * @param orderId order ID
     * @return OrderDTO
     */
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
}