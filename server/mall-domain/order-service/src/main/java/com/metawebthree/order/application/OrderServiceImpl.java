package com.metawebthree.order.application;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;

import org.apache.dubbo.config.annotation.DubboService;
import org.springframework.beans.factory.annotation.Autowired;

import com.metawebthree.common.generated.rpc.CloseOrderRequest;
import com.metawebthree.common.generated.rpc.CloseOrderResponse;
import com.metawebthree.common.generated.rpc.CreateReturnApplyRequest;
import com.metawebthree.common.generated.rpc.CreateReturnApplyResponse;
import com.metawebthree.common.generated.rpc.GetOrderByUserIdRequest;
import com.metawebthree.common.generated.rpc.GetOrderByUserIdResponse;
import com.metawebthree.common.generated.rpc.OrderDTO;
import com.metawebthree.common.generated.rpc.OrderService;
import com.metawebthree.common.generated.rpc.PaySuccessRequest;
import com.metawebthree.common.generated.rpc.PaySuccessResponse;
import com.metawebthree.common.generated.rpc.QueryLogisticsRequest;
import com.metawebthree.common.generated.rpc.QueryLogisticsResponse;
import com.metawebthree.common.generated.rpc.google.type.Money;
import com.metawebthree.order.domain.model.OrderDO;
import com.metawebthree.order.infrastructure.persistence.mapper.OrderMapper;

import lombok.extern.slf4j.Slf4j;

@Slf4j
@DubboService
public class OrderServiceImpl implements OrderService {

    @Autowired
    private OrderMapper orderMapper;

    @Autowired
    private OrderApplicationService orderApplicationService;

    @Override
    public GetOrderByUserIdResponse getOrderByUserId(GetOrderByUserIdRequest request) {
        Long userId = request.getId();
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

    private OrderDTO toDto(OrderDO o) {
        Money money = toMoney(o.getOrderAmount(), "USD");
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
        BigDecimal nanosDecimal = amount.subtract(BigDecimal.valueOf(units)).multiply(BigDecimal.valueOf(1_000_000_000L));
        int nanos = nanosDecimal.intValue();
        return Money.newBuilder().setCurrencyCode(currency).setUnits(units).setNanos(nanos).build();
    }

    @Override
    public CreateReturnApplyResponse createReturnApply(CreateReturnApplyRequest request) {
        log.info("Create return apply request received, orderId: {}", request.getOrderId());
        try {
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
        log.info("Query logistics request received, orderId: {}", request.getOrderId());
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
        log.info("Pay success request received, orderId: {}", request.getOrderId());
        try {
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
        } catch (Exception e) {
            log.error("Failed to process payment", e);
            return PaySuccessResponse.newBuilder()
                    .setSuccess(false)
                    .setMessage("Failed to process payment: " + e.getMessage())
                    .build();
        }
    }

    @Override
    public CompletableFuture<PaySuccessResponse> paySuccessAsync(PaySuccessRequest request) {
        return CompletableFuture.completedFuture(paySuccess(request));
    }

    @Override
    public CloseOrderResponse closeOrder(CloseOrderRequest request) {
        log.info("Close order request received, orderIds: {}", request.getIds());
        try {
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

}
