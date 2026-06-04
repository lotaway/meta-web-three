package com.metawebthree.order.interfaces.rpc;

import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.common.constants.PaginationConstants;
import com.metawebthree.common.generated.rpc.*;
import com.metawebthree.common.generated.rpc.google.type.Money;
import com.metawebthree.common.utils.ValidationUtils;
import com.metawebthree.order.application.AdminOrderQueryService;
import com.metawebthree.order.domain.model.OrderDO;
import com.metawebthree.order.domain.model.OrderItemDO;
import com.metawebthree.order.domain.model.ReturnApplyDO;
import com.metawebthree.order.infrastructure.persistence.mapper.OrderItemMapper;
import com.metawebthree.order.infrastructure.persistence.mapper.OrderMapper;
import com.metawebthree.order.infrastructure.persistence.mapper.ReturnApplyMapper;
import lombok.extern.slf4j.Slf4j;
import org.apache.dubbo.config.annotation.DubboService;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.stream.Collectors;

@DubboService
@Component
@Slf4j
public class OrderRpcService implements OrderService {

    private static final String DEFAULT_CURRENCY = "USD";
    private static final long NANOS_PER_UNIT = 1_000_000_000L;

    private final AdminOrderQueryService queryService;
    private final OrderMapper orderMapper;
    private final OrderItemMapper orderItemMapper;
    private final ReturnApplyMapper returnApplyMapper;

    public OrderRpcService(AdminOrderQueryService queryService,
                           OrderMapper orderMapper,
                           OrderItemMapper orderItemMapper,
                           ReturnApplyMapper returnApplyMapper) {
        this.queryService = queryService;
        this.orderMapper = orderMapper;
        this.orderItemMapper = orderItemMapper;
        this.returnApplyMapper = returnApplyMapper;
    }

    @Override
    public GetOrderStatusDistributionResponse getOrderStatusDistribution(GetOrderStatusDistributionRequest request) {
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
        try {
            int limit = request.getLimit() > 0 ? request.getLimit() : 10;
            List<Map<String, Object>> results = queryService.getHotProducts(limit);
            List<HotProductInfo> products = toHotProductInfos(results);
            return GetHotProductsResponse.newBuilder().addAllProducts(products).build();
        } catch (Exception e) {
            log.error("Failed to get hot products", e);
            return GetHotProductsResponse.newBuilder().build();
        }
    }

    private List<HotProductInfo> toHotProductInfos(List<Map<String, Object>> results) {
        return results.stream()
                .map(row -> HotProductInfo.newBuilder()
                        .setProductId(getLongValue(row, "productId"))
                        .setProductName(getStringValue(row, "productName"))
                        .setSalesCount(getLongValue(row, "salesCount"))
                        .setSalesAmount(getLongValue(row, "salesAmount"))
                        .build())
                .collect(Collectors.toList());
    }

    @Override
    public GetSalesByHourTodayResponse getSalesByHourToday(GetSalesByHourTodayRequest request) {
        try {
            List<Map<String, Object>> results = queryService.getSalesByHourToday();
            List<SalesByHourInfo> hourlyData = toSalesByHourInfos(results);
            return GetSalesByHourTodayResponse.newBuilder().addAllHourlyData(hourlyData).build();
        } catch (Exception e) {
            log.error("Failed to get sales by hour today", e);
            return GetSalesByHourTodayResponse.newBuilder().build();
        }
    }

    private List<SalesByHourInfo> toSalesByHourInfos(List<Map<String, Object>> results) {
        return results.stream()
                .map(row -> SalesByHourInfo.newBuilder()
                        .setHour(getIntValue(row, "hour"))
                        .setSales(getLongValue(row, "sales"))
                        .setOrders(getIntValue(row, "orders"))
                        .build())
                .collect(Collectors.toList());
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
        try {
            OrderDO order = orderMapper.selectById(request.getOrderId());
            if (order == null) {
                return CreateReturnApplyResponse.newBuilder()
                        .setSuccess(false)
                        .setMessage("Order not found")
                        .build();
            }
            ReturnApplyDO returnApply = new ReturnApplyDO();
            returnApply.setOrderId(request.getOrderId());
            returnApply.setReason(request.getReason());
            returnApply.setStatus(0);
            returnApply.setCreateTime(LocalDateTime.now());
            returnApplyMapper.insert(returnApply);
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
        try {
            OrderDO order = orderMapper.selectById(request.getOrderId());
            if (order == null) {
                return QueryLogisticsResponse.newBuilder().build();
            }
            return QueryLogisticsResponse.newBuilder()
                    .setLogisticsCompany("SF Express")
                    .setTrackingNumber("SF" + request.getOrderId())
                    .addTraces("Order created")
                    .addTraces("Shipped")
                    .addTraces("In transit")
                    .build();
        } catch (Exception e) {
            log.error("Failed to query logistics", e);
            return QueryLogisticsResponse.newBuilder().build();
        }
    }

    @Override
    public CompletableFuture<QueryLogisticsResponse> queryLogisticsAsync(QueryLogisticsRequest request) {
        return CompletableFuture.completedFuture(queryLogistics(request));
    }

    @Override
    public GetOrderByOrderNoResponse getOrderByOrderNo(GetOrderByOrderNoRequest request) {
        try {
            OrderDO order = orderMapper.selectOne(
                    new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<OrderDO>()
                            .eq(OrderDO::getOrderNo, request.getOrderNo()));
            if (order == null) {
                return GetOrderByOrderNoResponse.newBuilder().build();
            }
            return GetOrderByOrderNoResponse.newBuilder().setOrder(toDto(order)).build();
        } catch (Exception e) {
            log.error("Failed to get order by orderNo", e);
            return GetOrderByOrderNoResponse.newBuilder().build();
        }
    }

    @Override
    public CompletableFuture<GetOrderByOrderNoResponse> getOrderByOrderNoAsync(GetOrderByOrderNoRequest request) {
        return CompletableFuture.completedFuture(getOrderByOrderNo(request));
    }

    @Override
    public ListOrdersResponse listOrders(ListOrdersRequest request) {
        try {
            int page = request.getPage() > 0 ? request.getPage() : PaginationConstants.DEFAULT_PAGE;
            int size = request.getSize() > 0 ? request.getSize() : PaginationConstants.DEFAULT_SIZE;
            Page<OrderDO> mpPage = new Page<>(page, size);
            com.baomidou.mybatisplus.core.metadata.IPage<OrderDO> pageResult = orderMapper.selectPage(mpPage,
                    new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<OrderDO>()
                            .orderByDesc(OrderDO::getCreatedAt));
            List<OrderDTO> dtos = pageResult.getRecords().stream().map(this::toDto).toList();
            return ListOrdersResponse.newBuilder()
                    .setPage(page)
                    .setSize(size)
                    .setTotal(pageResult.getTotal())
                    .addAllOrders(dtos)
                    .build();
        } catch (Exception e) {
            log.error("Failed to list orders", e);
            return ListOrdersResponse.newBuilder().build();
        }
    }

    @Override
    public CompletableFuture<ListOrdersResponse> listOrdersAsync(ListOrdersRequest request) {
        return CompletableFuture.completedFuture(listOrders(request));
    }

    @Override
    public GetOrderByUserIdResponse getOrderByUserId(GetOrderByUserIdRequest request) {
        try {
            List<OrderDO> orders = orderMapper.selectList(
                    new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<OrderDO>()
                            .eq(OrderDO::getUserId, request.getUserId())
                            .orderByDesc(OrderDO::getCreatedAt));
            List<OrderDTO> dtos = orders.stream().map(this::toDto).toList();
            return GetOrderByUserIdResponse.newBuilder().addAllOrders(dtos).build();
        } catch (Exception e) {
            log.error("Failed to get orders by userId", e);
            return GetOrderByUserIdResponse.newBuilder().build();
        }
    }

    @Override
    public CompletableFuture<GetOrderByUserIdResponse> getOrderByUserIdAsync(GetOrderByUserIdRequest request) {
        return CompletableFuture.completedFuture(getOrderByUserId(request));
    }

    @Override
    public PaySuccessResponse paySuccess(PaySuccessRequest request) {
        try {
            OrderDO order = orderMapper.selectById(request.getOrderId());
            if (order == null) {
                return PaySuccessResponse.newBuilder()
                        .setSuccess(false)
                        .setMessage("Order not found")
                        .build();
            }
            order.setOrderStatus("PAID");
            order.setPaymentTime(LocalDateTime.now());
            orderMapper.updateById(order);
            return PaySuccessResponse.newBuilder()
                    .setSuccess(true)
                    .setMessage("Payment processed successfully")
                    .build();
        } catch (Exception e) {
            log.error("Failed to process payment", e);
            return PaySuccessResponse.newBuilder()
                    .setSuccess(false)
                    .setMessage("Payment failed: " + e.getMessage())
                    .build();
        }
    }

    @Override
    public CompletableFuture<PaySuccessResponse> paySuccessAsync(PaySuccessRequest request) {
        return CompletableFuture.completedFuture(paySuccess(request));
    }

    @Override
    public CloseOrderResponse closeOrder(CloseOrderRequest request) {
        try {
            String ids = request.getIds();
            if (ids == null || ids.isEmpty()) {
                return CloseOrderResponse.newBuilder()
                        .setSuccess(false)
                        .setMessage("Order ids is empty")
                        .build();
            }
            List<Long> orderIds = Arrays.stream(ids.split(","))
                    .map(String::trim)
                    .map(Long::parseLong)
                    .toList();
            for (Long orderId : orderIds) {
                OrderDO order = orderMapper.selectById(orderId);
                if (order != null) {
                    order.setOrderStatus("CLOSED");
                    orderMapper.updateById(order);
                }
            }
            return CloseOrderResponse.newBuilder()
                    .setSuccess(true)
                    .setMessage("Order closed successfully")
                    .build();
        } catch (Exception e) {
            log.error("Failed to close order", e);
            return CloseOrderResponse.newBuilder()
                    .setSuccess(false)
                    .setMessage("Failed to close order: " + e.getMessage())
                    .build();
        }
    }

    @Override
    public CompletableFuture<CloseOrderResponse> closeOrderAsync(CloseOrderRequest request) {
        return CompletableFuture.completedFuture(closeOrder(request));
    }

    @Override
    public CreateOrderResponse createOrder(CreateOrderRequest request) {
        try {
            if (request == null) {
                return CreateOrderResponse.newBuilder().setSuccess(false).setMessage("Request must not be null").build();
            }
            if (request.getItemsList().isEmpty()) {
                return CreateOrderResponse.newBuilder().setSuccess(false).setMessage("Order must have at least one item").build();
            }
            Long orderId = IdWorker.getId();
            String orderNo = String.valueOf(IdWorker.getId());
            BigDecimal total = BigDecimal.ZERO;
            for (OrderItemProto item : request.getItemsList()) {
                total = total.add(BigDecimal.valueOf(item.getPrice())
                        .multiply(BigDecimal.valueOf(item.getQuantity())));
            }
            OrderDO order = OrderDO.builder()
                    .id(orderId)
                    .userId(request.getUserId())
                    .orderNo(orderNo)
                    .orderStatus("CREATED")
                    .orderType("NORMAL")
                    .orderAmount(total)
                    .orderRemark(request.getOrderRemark())
                    .build();
            orderMapper.insert(order);
            for (OrderItemProto item : request.getItemsList()) {
                BigDecimal itemTotal = BigDecimal.valueOf(item.getPrice())
                        .multiply(BigDecimal.valueOf(item.getQuantity()));
                OrderItemDO orderItem = OrderItemDO.builder()
                        .id(IdWorker.getId())
                        .orderId(orderId)
                        .productId(item.getProductId())
                        .productName(item.getProductName())
                        .quantity(item.getQuantity())
                        .unitPrice(BigDecimal.valueOf(item.getPrice()))
                        .totalPrice(itemTotal)
                        .build();
                orderItemMapper.insert(orderItem);
            }
            return CreateOrderResponse.newBuilder()
                    .setOrderId(orderId)
                    .setOrderNo(orderNo)
                    .setSuccess(true)
                    .setMessage("Order created successfully")
                    .build();
        } catch (Exception e) {
            log.error("Failed to create order", e);
            return CreateOrderResponse.newBuilder()
                    .setSuccess(false)
                    .setMessage("Failed to create order: " + e.getMessage())
                    .build();
        }
    }

    @Override
    public CompletableFuture<CreateOrderResponse> createOrderAsync(CreateOrderRequest request) {
        return CompletableFuture.completedFuture(createOrder(request));
    }

    private OrderDTO toDto(OrderDO o) {
        Money money = toMoney(o.getOrderAmount(), DEFAULT_CURRENCY);
        return OrderDTO.newBuilder()
                .setId(o.getId())
                .setUserId(o.getUserId())
                .setOrderNo(o.getOrderNo())
                .setOrderStatus(o.getOrderStatus())
                .setOrderType(o.getOrderType())
                .setOrderAmount(money)
                .setOrderRemark(o.getOrderRemark() == null ? "" : o.getOrderRemark())
                .build();
    }

    private Money toMoney(BigDecimal amount, String currency) {
        long units = amount.longValue();
        BigDecimal nanosDecimal = amount.subtract(BigDecimal.valueOf(units)).multiply(BigDecimal.valueOf(NANOS_PER_UNIT));
        int nanos = ValidationUtils.safeIntFromLong(nanosDecimal.longValue(), "nanos");
        return Money.newBuilder().setCurrencyCode(currency).setUnits(units).setNanos(nanos).build();
    }

    private Long getLongValue(Map<String, Object> row, String key) {
        Object value = row.get(key);
        if (value == null) return 0L;
        if (value instanceof Number) return ((Number) value).longValue();
        try { return Long.parseLong(value.toString()); } catch (NumberFormatException e) { return 0L; }
    }

    private Integer getIntValue(Map<String, Object> row, String key) {
        Object value = row.get(key);
        if (value == null) return 0;
        if (value instanceof Number) return ((Number) value).intValue();
        try { return Integer.parseInt(value.toString()); } catch (NumberFormatException e) { return 0; }
    }

    private String getStringValue(Map<String, Object> row, String key) {
        Object value = row.get(key);
        return value != null ? value.toString() : "";
    }
}
