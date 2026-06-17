package com.metawebthree.gateway.client;

import com.metawebthree.common.enums.PaymentMethod;
import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.common.exception.BusinessException;
import com.metawebthree.common.generated.rpc.*;
import com.metawebthree.common.utils.ValidationUtils;
import lombok.extern.slf4j.Slf4j;
import org.apache.dubbo.config.annotation.DubboReference;
import org.springframework.stereotype.Component;

import java.util.*;

@Slf4j
@Component
public class OrderClient {

    @DubboReference(check = false, lazy = true)
    private OrderService orderService;

    public Map<String, Object> getOrderById(String id) {
        try {
            GetOrderByUserIdRequest request = GetOrderByUserIdRequest.newBuilder()
                    .setUserId(ValidationUtils.parseLong(id, "id"))
                    .build();
            GetOrderByUserIdResponse response = orderService.getOrderByUserId(request);
            List<OrderDTO> orders = response.getOrdersList();
            return buildOrderMap(orders.isEmpty() ? null : orders.get(0));
        } catch (Exception e) {
            log.error("Failed to get order by id: {}, error: {}", id, e.getMessage());
            throw new BusinessException(ResponseStatus.SYSTEM_ERROR, "Failed to get order by id: " + id, e);
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
            throw new BusinessException(ResponseStatus.SYSTEM_ERROR, "Failed to get order by orderNo: " + orderNo, e);
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
            throw new BusinessException(ResponseStatus.SYSTEM_ERROR, "Failed to list orders", e);
        }
    }

    public Map<String, Object> createOrder(Map<String, Object> input) {
        try {
            if (input == null) throw new IllegalArgumentException("input must not be null");
            CreateOrderRequest.Builder builder = CreateOrderRequest.newBuilder();
            Object userId = input.get("userId");
            if (userId != null) builder.setUserId(ValidationUtils.parseLongSafe(userId, "userId"));
            Object shippingAddress = input.get("shippingAddress");
            if (shippingAddress instanceof String s) builder.setShippingAddress(s);
            Object paymentMethod = input.get("paymentMethod");
            if (paymentMethod instanceof String s) builder.setPaymentMethod(s);
            Object orderRemark = input.get("orderRemark");
            if (orderRemark instanceof String s) builder.setOrderRemark(s);

            parseOrderItems(input.get("items"), builder);

            CreateOrderResponse response = orderService.createOrder(builder.build());
            return buildCreateOrderResult(response);
        } catch (Exception e) {
            log.error("Failed to create order, error: {}", e.getMessage());
            throw new BusinessException(ResponseStatus.SYSTEM_ERROR, "Failed to create order", e);
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
            throw new BusinessException(ResponseStatus.SYSTEM_ERROR, "Failed to cancel order: " + id, e);
        }
    }

    public boolean payOrder(String id, String paymentMethod) {
        try {
            if (paymentMethod == null) {
                throw new BusinessException(ResponseStatus.PARAM_VALIDATION_ERROR, "paymentMethod is required");
            }
            PaymentMethod method = switch (paymentMethod.toLowerCase()) {
                case "wechat", "weixin" -> PaymentMethod.WECHAT;
                case "alipay", "zhifubao" -> PaymentMethod.ALIPAY;
                case "card", "bankcard" -> PaymentMethod.CARD;
                default -> PaymentMethod.OTHER;
            };
            int payType = method.getCode();
            PaySuccessRequest request = PaySuccessRequest.newBuilder()
                    .setOrderId(ValidationUtils.parseLong(id, "id"))
                    .setPayType(payType)
                    .build();
            PaySuccessResponse response = orderService.paySuccess(request);
            return response.getSuccess();
        } catch (Exception e) {
            log.error("Failed to pay order: id={}, error: {}", id, e.getMessage());
            throw new BusinessException(ResponseStatus.SYSTEM_ERROR, "Failed to pay order: " + id, e);
        }
    }

    private void parseOrderItems(Object itemsObj, CreateOrderRequest.Builder builder) {
        if (!(itemsObj instanceof List)) return;
        for (Object itemObj : (List<?>) itemsObj) {
            if (!(itemObj instanceof Map)) continue;
            Map<String, Object> itemMap = (Map<String, Object>) itemObj;
            OrderItemProto.Builder itemBuilder = OrderItemProto.newBuilder();
            Object productId = itemMap.get("productId");
            if (productId != null) itemBuilder.setProductId(ValidationUtils.parseLongSafe(productId, "productId"));
            Object productName = itemMap.get("productName");
            if (productName instanceof String s) itemBuilder.setProductName(s);
            Object quantity = itemMap.get("quantity");
            if (quantity != null) itemBuilder.setQuantity(ValidationUtils.parseInt(quantity, "quantity"));
            Object price = itemMap.get("price");
            if (price != null) itemBuilder.setPrice(((Number) price).doubleValue());
            Object sku = itemMap.get("sku");
            if (sku instanceof String s) itemBuilder.setSku(s);
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
