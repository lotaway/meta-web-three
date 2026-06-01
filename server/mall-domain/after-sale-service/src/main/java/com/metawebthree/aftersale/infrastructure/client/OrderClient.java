package com.metawebthree.aftersale.infrastructure.client;

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
                .setId(orderId)
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

    /**
     * Close order for refund
     * @param orderIds order IDs
     * @param note note
     * @return true if success
     */
    public boolean closeOrderForRefund(String orderIds, String note) {
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

    /**
     * Create return apply in order service
     * @param orderId order ID
     * @param reason reason
     * @param userId user ID
     * @return true if success
     */
    public boolean createReturnApply(Long orderId, String reason, Long userId) {
        CreateReturnApplyRequest request = CreateReturnApplyRequest.newBuilder()
                .setOrderId(orderId)
                .setReason(reason)
                .setUserId(userId)
                .build();
        
        try {
            CreateReturnApplyResponse response = orderService.createReturnApply(request);
            return response.getSuccess();
        } catch (Exception e) {
            throw new BusinessException(ResponseStatus.SYSTEM_ERROR, "Failed to create return apply: " + e.getMessage());
        }
    }
}