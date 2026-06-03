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
            List<OrderDTO> orders = response.getOrdersList();
            return buildOrderMap(orders.isEmpty() ? null : orders.get(0));
        } catch (Exception e) {
            log.error("Failed to get order by id: {}, error: {}", id, e.getMessage());
        }
        return new HashMap<>();
    }

    public Map<String, Object> getOrderByOrderNo(String orderNo) {
        try {
            GetOrderByUserIdRequest request = GetOrderByUserIdRequest.newBuilder().build();
            GetOrderByUserIdResponse response = orderService.getOrderByUserId(request);
            for (OrderDTO order : response.getOrdersList()) {
                if (orderNo.equals(order.getOrderNo())) {
                    return buildOrderMap(order);
                }
            }
        } catch (Exception e) {
            log.error("Failed to get order by orderNo: {}, error: {}", orderNo, e.getMessage());
        }
        return new HashMap<>();
    }

    private Map<String, Object> buildOrderMap(OrderDTO order) {
        if (order == null) return new HashMap<>();
        Map<String, Object> result = new HashMap<>();
        result.put("id", order.getId());
        result.put("orderNo", order.getOrderNo());
        result.put("status", order.getOrderStatus());
        result.put("totalAmount", order.getOrderAmount() != null ? order.getOrderAmount().getUnits() : 0);
        result.put("userId", order.getUserId());
        return result;
    }

    public Map<String, Object> getOrders(int page, int size) {
        try {
            GetOrderStatusDistributionRequest request = GetOrderStatusDistributionRequest.newBuilder().build();
            GetOrderStatusDistributionResponse response = orderService.getOrderStatusDistribution(request);

            GetPendingOrdersCountRequest countRequest = GetPendingOrdersCountRequest.newBuilder().build();
            GetPendingOrdersCountResponse countResponse = orderService.getPendingOrdersCount(countRequest);

            return buildOrdersConnection(
                buildOrderEdges(response.getDistributionMap()),
                countResponse.getCount(),
                page
            );
        } catch (Exception e) {
            log.error("Failed to get orders: page={}, size={}, error: {}", page, size, e.getMessage());
        }
        return createEmptyOrdersConnection(page, size);
    }

    public Map<String, Object> createOrder(Map<String, Object> input) {
        try {
            CreateOrderRequest.Builder builder = CreateOrderRequest.newBuilder();
            if (input.containsKey("userId")) builder.setUserId(((Number) input.get("userId")).longValue());
            if (input.containsKey("shippingAddress")) builder.setShippingAddress((String) input.get("shippingAddress"));
            if (input.containsKey("paymentMethod")) builder.setPaymentMethod((String) input.get("paymentMethod"));
            if (input.containsKey("orderRemark")) builder.setOrderRemark((String) input.get("orderRemark"));

            parseOrderItems(input.get("items"), builder);

            CreateOrderResponse response = orderService.createOrder(builder.build());
            return buildCreateOrderResult(response);
        } catch (Exception e) {
            log.error("Failed to create order, error: {}", e.getMessage());
        }
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

    private List<Map<String, Object>> buildOrderEdges(Map<Integer, Integer> distributionMap) {
        List<Map<String, Object>> edges = new ArrayList<>();
        distributionMap.forEach((status, count) -> {
            Map<String, Object> edge = new HashMap<>();
            Map<String, Object> node = new HashMap<>();
            node.put("status", status);
            node.put("count", count);
            edge.put("node", node);
            edges.add(edge);
        });
        return edges;
    }

    private Map<String, Object> buildOrdersConnection(List<Map<String, Object>> edges, long totalCount, int page) {
        Map<String, Object> connection = new HashMap<>();
        connection.put("edges", edges);
        connection.put("totalCount", totalCount);
        connection.put("pageInfo", Map.of(
            "hasNextPage", false,
            "hasPreviousPage", page > 0
        ));
        return connection;
    }

    private void parseOrderItems(Object itemsObj, CreateOrderRequest.Builder builder) {
        if (!(itemsObj instanceof List)) return;
        for (Object itemObj : (List<?>) itemsObj) {
            if (!(itemObj instanceof Map)) continue;
            Map<String, Object> itemMap = (Map<String, Object>) itemObj;
            OrderItemProto.Builder itemBuilder = OrderItemProto.newBuilder();
            if (itemMap.containsKey("productId")) itemBuilder.setProductId(((Number) itemMap.get("productId")).longValue());
            if (itemMap.containsKey("productName")) itemBuilder.setProductName((String) itemMap.get("productName"));
            if (itemMap.containsKey("quantity")) itemBuilder.setQuantity(((Number) itemMap.get("quantity")).intValue());
            if (itemMap.containsKey("price")) itemBuilder.setPrice(((Number) itemMap.get("price")).doubleValue());
            if (itemMap.containsKey("sku")) itemBuilder.setSku((String) itemMap.get("sku"));
            builder.addItems(itemBuilder.build());
        }
    }

    private Map<String, Object> buildCreateOrderResult(CreateOrderResponse response) {
        Map<String, Object> result = new HashMap<>();
        result.put("orderId", response.getOrderId());
        result.put("orderNo", response.getOrderNo());
        result.put("success", response.getSuccess());
        result.put("message", response.getMessage());
        return result;
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
