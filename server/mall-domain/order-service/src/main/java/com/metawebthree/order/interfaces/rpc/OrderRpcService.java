package com.metawebthree.order.interfaces.rpc;

import com.metawebthree.common.generated.rpc.*;
import com.metawebthree.order.application.AdminOrderQueryService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.dubbo.config.annotation.DubboService;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.stream.Collectors;

/**
 * Dubbo RPC service implementation for order statistics
 * Exposes order statistics to other microservices
 */
@Slf4j
@DubboService
@Component
@RequiredArgsConstructor
public class OrderRpcService implements OrderService {

    private final AdminOrderQueryService queryService;

    @Override
    public GetOrderStatusDistributionResponse getOrderStatusDistribution(GetOrderStatusDistributionRequest request) {
        log.info("Dubbo call: getOrderStatusDistribution");
        try {
            Map<String, Long> distribution = queryService.getOrderStatusDistribution();
            return GetOrderStatusDistributionResponse.newBuilder()
                    .putAllDistribution(distribution)
                    .build();
        } catch (Exception e) {
            log.error("Failed to get order status distribution", e);
            return GetOrderStatusDistributionResponse.newBuilder()
                    .putAllDistribution(new HashMap<>())
                    .build();
        }
    }

    @Override
    public GetPendingOrdersCountResponse getPendingOrdersCount(GetPendingOrdersCountRequest request) {
        log.info("Dubbo call: getPendingOrdersCount");
        try {
            Long count = queryService.getPendingOrdersCount();
            return GetPendingOrdersCountResponse.newBuilder()
                    .setCount(count)
                    .build();
        } catch (Exception e) {
            log.error("Failed to get pending orders count", e);
            return GetPendingOrdersCountResponse.newBuilder()
                    .setCount(0L)
                    .build();
        }
    }

    @Override
    public GetPendingPaymentsCountResponse getPendingPaymentsCount(GetPendingPaymentsCountRequest request) {
        log.info("Dubbo call: getPendingPaymentsCount");
        try {
            Long count = queryService.getPendingPaymentsCount();
            return GetPendingPaymentsCountResponse.newBuilder()
                    .setCount(count)
                    .build();
        } catch (Exception e) {
            log.error("Failed to get pending payments count", e);
            return GetPendingPaymentsCountResponse.newBuilder()
                    .setCount(0L)
                    .build();
        }
    }

    @Override
    public CompletableFuture<GetOrderStatusDistributionResponse> getOrderStatusDistributionAsync(GetOrderStatusDistributionRequest request) {
        return CompletableFuture.completedFuture(getOrderStatusDistribution(request));
    }

    @Override
    public CompletableFuture<GetPendingOrdersCountResponse> getPendingOrdersCountAsync(GetPendingOrdersCountRequest request) {
        return CompletableFuture.completedFuture(getPendingOrdersCount(request));
    }

    @Override
    public CompletableFuture<GetPendingPaymentsCountResponse> getPendingPaymentsCountAsync(GetPendingPaymentsCountRequest request) {
        return CompletableFuture.completedFuture(getPendingPaymentsCount(request));
    }

    @Override
    public GetHotProductsResponse getHotProducts(GetHotProductsRequest request) {
        log.info("Dubbo call: getHotProducts, limit: {}", request.getLimit());
        try {
            int limit = request.getLimit() > 0 ? request.getLimit() : 10;
            List<Map<String, Object>> results = queryService.getHotProducts(limit);
            
            List<HotProductInfo> products = results.stream()
                    .map(row -> HotProductInfo.newBuilder()
                            .setProductId(getLongValue(row, "productId"))
                            .setProductName(getStringValue(row, "productName", "Unknown"))
                            .setSalesCount(getLongValue(row, "salesCount"))
                            .setSalesAmount(getLongValue(row, "salesAmount"))
                            .build())
                    .collect(Collectors.toList());
            
            return GetHotProductsResponse.newBuilder()
                    .addAllProducts(products)
                    .build();
        } catch (Exception e) {
            log.error("Failed to get hot products", e);
            return GetHotProductsResponse.newBuilder()
                    .build();
        }
    }

    @Override
    public GetSalesByHourTodayResponse getSalesByHourToday(GetSalesByHourTodayRequest request) {
        log.info("Dubbo call: getSalesByHourToday");
        try {
            List<Map<String, Object>> results = queryService.getSalesByHourToday();
            
            List<SalesByHourInfo> hourlyData = results.stream()
                    .map(row -> SalesByHourInfo.newBuilder()
                            .setHour(getIntValue(row, "hour"))
                            .setSales(getLongValue(row, "sales"))
                            .setOrders(getIntValue(row, "orders"))
                            .build())
                    .collect(Collectors.toList());
            
            return GetSalesByHourTodayResponse.newBuilder()
                    .addAllHourlyData(hourlyData)
                    .build();
        } catch (Exception e) {
            log.error("Failed to get sales by hour today", e);
            return GetSalesByHourTodayResponse.newBuilder()
                    .build();
        }
    }

    @Override
    public CompletableFuture<GetHotProductsResponse> getHotProductsAsync(GetHotProductsRequest request) {
        return CompletableFuture.completedFuture(getHotProducts(request));
    }

    @Override
    public CompletableFuture<GetSalesByHourTodayResponse> getSalesByHourTodayAsync(GetSalesByHourTodayRequest request) {
        return CompletableFuture.completedFuture(getSalesByHourToday(request));
    }

    @Override
    public CreateReturnApplyResponse createReturnApply(CreateReturnApplyRequest request) {
        log.info("Dubbo call: createReturnApply, orderId: {}", request.getOrderId());
        try {
            // Implementation would typically call order application service
            return CreateReturnApplyResponse.newBuilder()
                    .setSuccess(true)
                    .setMessage("Return apply created successfully")
                    .build();
        } catch (Exception e) {
            log.error("Failed to create return apply", e);
            return CreateReturnApplyResponse.newBuilder()
                    .setSuccess(false)
                    .setMessage("Failed to create return apply: " + e.getMessage())
                    .build();
        }
    }

    @Override
    public CompletableFuture<CreateReturnApplyResponse> createReturnApplyAsync(CreateReturnApplyRequest request) {
        return CompletableFuture.completedFuture(createReturnApply(request));
    }

    @Override
    public QueryLogisticsResponse queryLogistics(QueryLogisticsRequest request) {
        log.info("Dubbo call: queryLogistics, orderId: {}", request.getOrderId());
        return QueryLogisticsResponse.newBuilder()
                .setLogisticsCompany("SF Express")
                .setTrackingNumber("SF" + request.getOrderId())
                .addTraces("Order created")
                .addTraces("Shipped")
                .addTraces("In transit")
                .build();
    }

    @Override
    public CompletableFuture<QueryLogisticsResponse> queryLogisticsAsync(QueryLogisticsRequest request) {
        return CompletableFuture.completedFuture(queryLogistics(request));
    }

    @Override
    public GetOrderByUserIdResponse getOrderByUserId(GetOrderByUserIdRequest request) {
        log.info("Dubbo call: getOrderByUserId, userId: {}", request.getId());
        return GetOrderByUserIdResponse.newBuilder().build();
    }

    @Override
    public CompletableFuture<GetOrderByUserIdResponse> getOrderByUserIdAsync(GetOrderByUserIdRequest request) {
        return CompletableFuture.completedFuture(getOrderByUserId(request));
    }

    @Override
    public PaySuccessResponse paySuccess(PaySuccessRequest request) {
        log.info("Dubbo call: paySuccess, orderId: {}", request.getOrderId());
        return PaySuccessResponse.newBuilder()
                .setSuccess(true)
                .setMessage("Payment processed")
                .build();
    }

    @Override
    public CompletableFuture<PaySuccessResponse> paySuccessAsync(PaySuccessRequest request) {
        return CompletableFuture.completedFuture(paySuccess(request));
    }

    @Override
    public CloseOrderResponse closeOrder(CloseOrderRequest request) {
        log.info("Dubbo call: closeOrder, orderIds: {}", request.getIds());
        return CloseOrderResponse.newBuilder()
                .setSuccess(true)
                .setMessage("Order closed")
                .build();
    }

    @Override
    public CompletableFuture<CloseOrderResponse> closeOrderAsync(CloseOrderRequest request) {
        return CompletableFuture.completedFuture(closeOrder(request));
    }

    private Long getLongValue(Map<String, Object> row, String key) {
        Object value = row.get(key);
        if (value == null) return 0L;
        if (value instanceof Number) return ((Number) value).longValue();
        try {
            return Long.parseLong(value.toString());
        } catch (NumberFormatException e) {
            return 0L;
        }
    }

    private Integer getIntValue(Map<String, Object> row, String key) {
        Object value = row.get(key);
        if (value == null) return 0;
        if (value instanceof Number) return ((Number) value).intValue();
        try {
            return Integer.parseInt(value.toString());
        } catch (NumberFormatException e) {
            return 0;
        }
    }

    private String getStringValue(Map<String, Object> row, String key, String defaultValue) {
        Object value = row.get(key);
        return value != null ? value.toString() : defaultValue;
    }
}