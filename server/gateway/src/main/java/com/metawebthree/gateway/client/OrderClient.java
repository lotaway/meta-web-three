package com.metawebthree.gateway.client;

import com.metawebthree.common.generated.rpc.*;
import lombok.extern.slf4j.Slf4j;
import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.stereotype.Component;

import java.util.*;

@Slf4j
@Component
public class OrderClient {

    @DubboReference
    private OrderService orderService;

    public Map<String, Object> getOrderById(String id) {
        try {
            GetOrderByUserIdRequest request = GetOrderByUserIdRequest.newBuilder()
                    .setId(Long.parseLong(id))
                    .build();
            GetOrderByUserIdResponse response = orderService.getOrderByUserId(request);

            Map<String, Object> result = new HashMap<>();
            List<OrderDTO> orders = response.getOrdersList();
            OrderDTO order = orders.isEmpty() ? null : orders.get(0);
            if (order != null) {
                result.put("id", order.getId());
                result.put("orderNo", order.getOrderNo());
                result.put("status", order.getOrderStatus());
                result.put("totalAmount", order.getOrderAmount() != null ? order.getOrderAmount().getUnits() : 0);
                result.put("userId", order.getUserId());
            }
            return result;
        } catch (Exception e) {
            log.error("Failed to get order by id: {}, error: {}", id, e.getMessage());
        }
        return new HashMap<>();
    }

    public Map<String, Object> getOrderByOrderNo(String orderNo) {
        try {
            GetOrderByUserIdRequest request = GetOrderByUserIdRequest.newBuilder().build();
            GetOrderByUserIdResponse response = orderService.getOrderByUserId(request);

            Map<String, Object> result = new HashMap<>();
            for (OrderDTO order : response.getOrdersList()) {
                if (orderNo.equals(order.getOrderNo())) {
                    result.put("id", order.getId());
                    result.put("orderNo", order.getOrderNo());
                    result.put("status", order.getOrderStatus());
                    result.put("totalAmount", order.getOrderAmount() != null ? order.getOrderAmount().getUnits() : 0);
                    break;
                }
            }
            return result;
        } catch (Exception e) {
            log.error("Failed to get order by orderNo: {}, error: {}", orderNo, e.getMessage());
        }
        return new HashMap<>();
    }

    public Map<String, Object> getOrders(int page, int size) {
        try {
            GetOrderStatusDistributionRequest request = GetOrderStatusDistributionRequest.newBuilder().build();
            GetOrderStatusDistributionResponse response = orderService.getOrderStatusDistribution(request);

            Map<String, Object> connection = new HashMap<>();
            List<Map<String, Object>> edges = new ArrayList<>();

            response.getDistributionMap().forEach((status, count) -> {
                Map<String, Object> edge = new HashMap<>();
                Map<String, Object> node = new HashMap<>();
                node.put("status", status);
                node.put("count", count);
                edge.put("node", node);
                edges.add(edge);
            });

            GetPendingOrdersCountRequest countRequest = GetPendingOrdersCountRequest.newBuilder().build();
            GetPendingOrdersCountResponse countResponse = orderService.getPendingOrdersCount(countRequest);

            connection.put("edges", edges);
            connection.put("totalCount", countResponse.getCount());
            connection.put("pageInfo", Map.of(
                "hasNextPage", false,
                "hasPreviousPage", page > 0
            ));
            return connection;
        } catch (Exception e) {
            log.error("Failed to get orders: page={}, size={}, error: {}", page, size, e.getMessage());
        }
        return createEmptyOrdersConnection(page, size);
    }

    public Map<String, Object> createOrder(Map<String, Object> input) {
        log.warn("createOrder via Dubbo not implemented - requires REST fallback to POST /api/orders with create order proto/RPC");
        return new HashMap<>();
    }

    public boolean cancelOrder(String id) {
        try {
            CloseOrderRequest request = CloseOrderRequest.newBuilder()
                    .setIds(id)
                    .build();
            CloseOrderResponse response = orderService.closeOrder(request);
            return response.getSuccess();
        } catch (Exception e) {
            log.error("Failed to cancel order: id={}, error: {}", id, e.getMessage());
        }
        return false;
    }

    public boolean payOrder(String id, String paymentMethod) {
        try {
            int payType = switch (paymentMethod.toLowerCase()) {
                case "wechat", "weixin" -> 1;
                case "alipay", "zhifubao" -> 2;
                case "card", "bankcard" -> 3;
                default -> 4;
            };
            PaySuccessRequest request = PaySuccessRequest.newBuilder()
                    .setOrderId(Long.parseLong(id))
                    .setPayType(payType)
                    .build();
            PaySuccessResponse response = orderService.paySuccess(request);
            return response.getSuccess();
        } catch (Exception e) {
            log.error("Failed to pay order: id={}, error: {}", id, e.getMessage());
        }
        return false;
    }

    private Map<String, Object> createEmptyOrdersConnection(int page, int size) {
        Map<String, Object> connection = new HashMap<>();
        connection.put("edges", new ArrayList<>());
        connection.put("totalCount", 0);
        connection.put("pageInfo", Map.of(
            "hasNextPage", false,
            "hasPreviousPage", page > 0
        ));
        return connection;
    }
}
