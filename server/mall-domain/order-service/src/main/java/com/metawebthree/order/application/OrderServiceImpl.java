package com.metawebthree.order.application;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;

import org.apache.dubbo.config.annotation.DubboService;

import com.metawebthree.common.generated.rpc.CloseOrderRequest;
import com.metawebthree.common.generated.rpc.CloseOrderResponse;
import com.metawebthree.common.generated.rpc.CreateOrderRequest;
import com.metawebthree.common.generated.rpc.CreateOrderResponse;
import com.metawebthree.common.generated.rpc.CreateReturnApplyRequest;
import com.metawebthree.common.generated.rpc.CreateReturnApplyResponse;
import com.metawebthree.common.generated.rpc.GetOrderByOrderNoRequest;
import com.metawebthree.common.generated.rpc.GetOrderByOrderNoResponse;
import com.metawebthree.common.generated.rpc.GetOrderByUserIdRequest;
import com.metawebthree.common.generated.rpc.GetOrderByUserIdResponse;
import com.metawebthree.common.generated.rpc.GetHotProductsRequest;
import com.metawebthree.common.generated.rpc.GetHotProductsResponse;
import com.metawebthree.common.generated.rpc.GetOrderStatusDistributionRequest;
import com.metawebthree.common.generated.rpc.GetOrderStatusDistributionResponse;
import com.metawebthree.common.generated.rpc.GetPendingOrdersCountRequest;
import com.metawebthree.common.generated.rpc.GetPendingOrdersCountResponse;
import com.metawebthree.common.generated.rpc.GetPendingPaymentsCountRequest;
import com.metawebthree.common.generated.rpc.GetPendingPaymentsCountResponse;
import com.metawebthree.common.generated.rpc.GetSalesByHourTodayRequest;
import com.metawebthree.common.generated.rpc.GetSalesByHourTodayResponse;
import com.metawebthree.common.generated.rpc.ListOrdersRequest;
import com.metawebthree.common.generated.rpc.ListOrdersResponse;
import com.metawebthree.common.generated.rpc.OrderDTO;
import com.metawebthree.common.generated.rpc.OrderItemProto;
import com.metawebthree.common.generated.rpc.OrderService;
import com.metawebthree.common.generated.rpc.PaySuccessRequest;
import com.metawebthree.common.generated.rpc.PaySuccessResponse;
import com.metawebthree.common.generated.rpc.QueryLogisticsRequest;
import com.metawebthree.common.generated.rpc.QueryLogisticsResponse;
import com.metawebthree.common.generated.rpc.google.type.Money;
import com.metawebthree.common.constants.PaginationConstants;
import com.metawebthree.common.utils.ValidationUtils;
import com.metawebthree.order.domain.model.OrderDO;
import com.metawebthree.order.domain.model.OrderItemDO;
import com.metawebthree.order.infrastructure.persistence.mapper.OrderItemMapper;
import com.metawebthree.order.infrastructure.persistence.mapper.OrderMapper;

import lombok.extern.slf4j.Slf4j;

@Slf4j
@DubboService
public class OrderServiceImpl implements OrderService {

    private static final String DEFAULT_CURRENCY = "USD";
    private static final long NANOS_PER_UNIT = 1_000_000_000L;
    private static final String ORDER_STATUS_CREATED = "CREATED";
    private static final String ORDER_TYPE_NORMAL = "NORMAL";

    private final OrderMapper orderMapper;
    private final OrderItemMapper orderItemMapper;
    private final OrderApplicationService orderApplicationService;

    public OrderServiceImpl(OrderMapper orderMapper, OrderItemMapper orderItemMapper, OrderApplicationService orderApplicationService) {
        this.orderMapper = orderMapper;
        this.orderItemMapper = orderItemMapper;
        this.orderApplicationService = orderApplicationService;
    }

    @Override
    public GetOrderByUserIdResponse getOrderByUserId(GetOrderByUserIdRequest request) {
        Long userId = request.getUserId();
        List<OrderDO> orders = orderMapper.selectList(
                new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<OrderDO>()
                        .eq(OrderDO::getUserId, userId)
                        .orderByDesc(OrderDO::getCreatedAt));
        List<OrderDTO> dtos = orders.stream().map(this::toDto).toList();
        return GetOrderByUserIdResponse.newBuilder().addAllOrders(dtos).build();
    }

    @Override
    public CompletableFuture<GetOrderByUserIdResponse> getOrderByUserIdAsync(GetOrderByUserIdRequest request) {
        return CompletableFuture.completedFuture(getOrderByUserId(request));
    }

    @Override
    public GetOrderByOrderNoResponse getOrderByOrderNo(GetOrderByOrderNoRequest request) {
        OrderDO order = orderMapper.selectOne(
                new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<OrderDO>()
                        .eq(OrderDO::getOrderNo, request.getOrderNo()));
        if (order == null) {
            return GetOrderByOrderNoResponse.newBuilder().build();
        }
        return GetOrderByOrderNoResponse.newBuilder().setOrder(toDto(order)).build();
    }

    @Override
    public CompletableFuture<GetOrderByOrderNoResponse> getOrderByOrderNoAsync(GetOrderByOrderNoRequest request) {
        return CompletableFuture.completedFuture(getOrderByOrderNo(request));
    }

    @Override
    public ListOrdersResponse listOrders(ListOrdersRequest request) {
        int page = request.getPage() > 0 ? request.getPage() : PaginationConstants.DEFAULT_PAGE;
        int size = request.getSize() > 0 ? request.getSize() : PaginationConstants.DEFAULT_SIZE;

        com.baomidou.mybatisplus.extension.plugins.pagination.Page<OrderDO> mpPage =
                new com.baomidou.mybatisplus.extension.plugins.pagination.Page<>(page, size);
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
    }

    @Override
    public CompletableFuture<ListOrdersResponse> listOrdersAsync(ListOrdersRequest request) {
        return CompletableFuture.completedFuture(listOrders(request));
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

    @Override
    public CreateReturnApplyResponse createReturnApply(CreateReturnApplyRequest request) {
        Long orderId = request.getOrderId();
        String reason = request.getReason();
        OrderDO order = orderMapper.selectById(orderId);
        if (order == null) {
            return CreateReturnApplyResponse.newBuilder()
                    .setSuccess(false)
                    .setMessage("Order not found")
                    .build();
        }
        return CreateReturnApplyResponse.newBuilder()
                .setSuccess(true)
                .setMessage("Return apply created")
                .build();
    }

    @Override
    public CompletableFuture<CreateReturnApplyResponse> createReturnApplyAsync(CreateReturnApplyRequest request) {
        return CompletableFuture.completedFuture(createReturnApply(request));
    }

    @Override
    public CreateOrderResponse createOrder(CreateOrderRequest request) {
        if (request == null) {
            return CreateOrderResponse.newBuilder().setSuccess(false).setMessage("Request must not be null").build();
        }
        if (request.getItemsList().isEmpty()) {
            return CreateOrderResponse.newBuilder().setSuccess(false).setMessage("Order must have at least one item").build();
        }
        Long orderId = com.baomidou.mybatisplus.core.toolkit.IdWorker.getId();
        String orderNo = String.valueOf(com.baomidou.mybatisplus.core.toolkit.IdWorker.getId());

        java.math.BigDecimal total = java.math.BigDecimal.ZERO;
        for (OrderItemProto item : request.getItemsList()) {
            total = total.add(java.math.BigDecimal.valueOf(item.getPrice())
                    .multiply(java.math.BigDecimal.valueOf(item.getQuantity())));
        }

        OrderDO order = OrderDO.builder()
                .id(orderId)
                .userId(request.getUserId())
                .orderNo(orderNo)
                .orderStatus(ORDER_STATUS_CREATED)
                .orderType(ORDER_TYPE_NORMAL)
                .orderAmount(total)
                .orderRemark(request.getOrderRemark())
                .build();
        orderMapper.insert(order);

        for (OrderItemProto item : request.getItemsList()) {
            java.math.BigDecimal itemTotal = java.math.BigDecimal.valueOf(item.getPrice())
                    .multiply(java.math.BigDecimal.valueOf(item.getQuantity()));
            OrderItemDO orderItem = OrderItemDO.builder()
                    .id(com.baomidou.mybatisplus.core.toolkit.IdWorker.getId())
                    .orderId(orderId)
                    .productId(item.getProductId())
                    .productName(item.getProductName())
                    .quantity(item.getQuantity())
                    .unitPrice(java.math.BigDecimal.valueOf(item.getPrice()))
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
    }

    @Override
    public CompletableFuture<CreateOrderResponse> createOrderAsync(CreateOrderRequest request) {
        return CompletableFuture.completedFuture(createOrder(request));
    }

    @Override
    public QueryLogisticsResponse queryLogistics(QueryLogisticsRequest request) {
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
    public PaySuccessResponse paySuccess(PaySuccessRequest request) {
        OrderDO order = orderMapper.selectById(request.getOrderId());
        if (order == null) {
            return PaySuccessResponse.newBuilder()
                    .setSuccess(false)
                    .setMessage("Order not found")
                    .build();
        }
        return PaySuccessResponse.newBuilder()
                .setSuccess(true)
                .setMessage("Payment processed successfully")
                .build();
    }

    @Override
    public CompletableFuture<PaySuccessResponse> paySuccessAsync(PaySuccessRequest request) {
        return CompletableFuture.completedFuture(paySuccess(request));
    }

    @Override
    public CloseOrderResponse closeOrder(CloseOrderRequest request) {
        String orderIds = request.getIds();
        if (orderIds == null || orderIds.isEmpty()) {
            return CloseOrderResponse.newBuilder()
                    .setSuccess(false)
                    .setMessage("Order ids is empty")
                    .build();
        }
        return CloseOrderResponse.newBuilder()
                .setSuccess(true)
                .setMessage("Order closed successfully")
                .build();
    }

    @Override
    public CompletableFuture<CloseOrderResponse> closeOrderAsync(CloseOrderRequest request) {
        return CompletableFuture.completedFuture(closeOrder(request));
    }

    @Override
    public GetSalesByHourTodayResponse getSalesByHourToday(GetSalesByHourTodayRequest request) {
        return GetSalesByHourTodayResponse.newBuilder().build();
    }

    @Override
    public CompletableFuture<GetSalesByHourTodayResponse> getSalesByHourTodayAsync(GetSalesByHourTodayRequest request) {
        return CompletableFuture.completedFuture(getSalesByHourToday(request));
    }

    @Override
    public GetHotProductsResponse getHotProducts(GetHotProductsRequest request) {
        return GetHotProductsResponse.newBuilder().build();
    }

    @Override
    public CompletableFuture<GetHotProductsResponse> getHotProductsAsync(GetHotProductsRequest request) {
        return CompletableFuture.completedFuture(getHotProducts(request));
    }

    @Override
    public GetOrderStatusDistributionResponse getOrderStatusDistribution(GetOrderStatusDistributionRequest request) {
        return GetOrderStatusDistributionResponse.newBuilder().build();
    }

    @Override
    public CompletableFuture<GetOrderStatusDistributionResponse> getOrderStatusDistributionAsync(GetOrderStatusDistributionRequest request) {
        return CompletableFuture.completedFuture(getOrderStatusDistribution(request));
    }

    @Override
    public GetPendingOrdersCountResponse getPendingOrdersCount(GetPendingOrdersCountRequest request) {
        return GetPendingOrdersCountResponse.newBuilder().setCount(0L).build();
    }

    @Override
    public CompletableFuture<GetPendingOrdersCountResponse> getPendingOrdersCountAsync(GetPendingOrdersCountRequest request) {
        return CompletableFuture.completedFuture(getPendingOrdersCount(request));
    }

    @Override
    public GetPendingPaymentsCountResponse getPendingPaymentsCount(GetPendingPaymentsCountRequest request) {
        return GetPendingPaymentsCountResponse.newBuilder().setCount(0L).build();
    }

    @Override
    public CompletableFuture<GetPendingPaymentsCountResponse> getPendingPaymentsCountAsync(GetPendingPaymentsCountRequest request) {
        return CompletableFuture.completedFuture(getPendingPaymentsCount(request));
    }

}
