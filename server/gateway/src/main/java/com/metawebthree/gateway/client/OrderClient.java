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

    /**
     * Get order by ID
     * @param id order ID
     * @return order data map
     */
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

    /**
     * Get order by order number
     * @param orderNo order number
     * @return order data map
     */
    public Map<String, Object> getOrderByOrderNo(String orderNo) {
        try {
            // GetOrderByUserIdRequest doesn't have setOrderNo, iterate through orders to find by orderNo
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

    /**
     * Get orders with pagination - uses statistics RPC
     * @param page page number
     * @param size page size
     * @return orders connection
     */
    public Map<String, Object> getOrders(int page, int size) {
        try {
            // Get order status distribution
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
            
            // Get total count
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

    /**
     * Create order - not implemented via Dubbo yet
     * @param input order input
     * @return created order
     */
    public Map<String, Object> createOrder(Map<String, Object> input) {
        log.warn("createOrder via Dubbo not implemented - requires REST fallback");
        return new HashMap<>();
    }

    /**
     * Cancel order
     * @param id order ID
     * @return true if success
     */
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

    /**
     * Pay order
     * @param id order ID
     * @param paymentMethod payment method
     * @return true if success
     */
    public boolean payOrder(String id, String paymentMethod) {
        try {
            // Map payment method to pay type (1=WeChat, 2=Alipay, 3=Card, 4=Balance)
            int payType = switch (paymentMethod.toLowerCase()) {
                case "wechat", "weixin" -> 1;
                case "alipay", "zhifubao" -> 2;
                case "card", "bankcard" -> 3;
                default -> 4; // balance
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