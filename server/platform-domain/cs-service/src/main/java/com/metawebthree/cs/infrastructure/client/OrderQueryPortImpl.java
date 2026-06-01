package com.metawebthree.cs.infrastructure.client;

import java.util.Optional;

import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.stereotype.Component;

import com.metawebthree.common.generated.rpc.*;
import com.metawebthree.cs.domain.ports.OrderQueryPort;

import lombok.extern.slf4j.Slf4j;

@Component
@Slf4j
public class OrderQueryPortImpl implements OrderQueryPort {

    @DubboReference(check = false, lazy = true)
    private OrderService orderService;

    @Override
    public Optional<String> findOrderStatus(Long orderId) {
        try {
            GetOrderByUserIdRequest request = GetOrderByUserIdRequest.newBuilder()
                    .setId(orderId)
                    .build();
            GetOrderByUserIdResponse response = orderService.getOrderByUserId(request);
            if (response != null && response.getOrdersCount() > 0) {
                return Optional.of(response.getOrders(0).getOrderStatus());
            }
        } catch (Exception e) {
            log.error("Query order status failed, orderId: {}, error: {}", orderId, e.getMessage());
        }
        return Optional.empty();
    }

    @Override
    public Optional<String> findOrderJson(Long orderId) {
        try {
            GetOrderByUserIdRequest request = GetOrderByUserIdRequest.newBuilder()
                    .setId(orderId)
                    .build();
            GetOrderByUserIdResponse response = orderService.getOrderByUserId(request);
            if (response != null && response.getOrdersCount() > 0) {
                return Optional.of(response.getOrders(0).toString());
            }
        } catch (Exception e) {
            log.error("Query order json failed, orderId: {}, error: {}", orderId, e.getMessage());
        }
        return Optional.empty();
    }
}