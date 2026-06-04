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
            throw new RuntimeException("Failed to get order by id: " + id, e);
        }
    }

    public Map<String, Object> getOrderByOrderNo(String orderNo) {
        try {
            GetOrderByOrderNoRequest request = GetOrderByOrderNoRequest.newBuilder()
                    .setOrderNo(orderNo)
                    .build();
            GetOrderByOrderNoResponse response = orderService.getOrderByOrderNo(request);
            return buildOrderMap(response.hasOrder() ? response.getOrder() : null);
        } catch (Exception e) {
            log.error("Failed to get order by orderNo: {}, error: {}", orderNo, e.getMessage());
            throw new RuntimeException("Failed to get order by orderNo: " + orderNo, e);
        }
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
            ListOrdersRequest request = ListOrdersRequest.newBuilder()
                    .setPage(page)
                    .setSize(size)
                    .build();
            ListOrdersResponse response = orderService.listOrders(request);
            Map<String, Object> result = new HashMap<>();
            result.put("page", response.getPage());
            result.put("size", response.getSize());
            result.put("total", response.getTotal());
            List<Map<String, Object>> orderMaps = response.getOrdersList().stream()
                    .map(this::buildOrderMap)
                    .toList();
            result.put("orders", orderMaps);
            return result;
        } catch (Exception e) {
            log.error("Failed to list orders, page: {}, size: {}, error: {}", page, size, e.getMessage());
            throw new RuntimeException("Failed to list orders", e);
        }
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
            throw new RuntimeException("Failed to create order", e);
        }
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
            throw new RuntimeException("Failed to cancel order: " + id, e);
        }
    }

    public boolean payOrder(String id, String paymentMethod) {
        try {
            if (paymentMethod == null) {
                throw new IllegalArgumentException("paymentMethod is required");
            }
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
            throw new RuntimeException("Failed to pay order: " + id, e);
        }
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
}
