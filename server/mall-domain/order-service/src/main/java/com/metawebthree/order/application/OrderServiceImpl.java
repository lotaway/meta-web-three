package com.metawebthree.order.application;

import java.math.BigDecimal;
import java.util.List;
import java.util.concurrent.CompletableFuture;

import org.apache.dubbo.config.annotation.DubboService;
import org.springframework.beans.factory.annotation.Autowired;

import com.metawebthree.common.generated.rpc.GetOrderByUserIdRequest;
import com.metawebthree.common.generated.rpc.GetOrderByUserIdResponse;
import com.metawebthree.common.generated.rpc.OrderDTO;
import com.metawebthree.common.generated.rpc.OrderService;
import com.metawebthree.common.generated.rpc.google.type.Money;
import com.metawebthree.order.domain.model.OrderDO;
import com.metawebthree.order.infrastructure.persistence.mapper.OrderMapper;

import lombok.extern.slf4j.Slf4j;

@Slf4j
@DubboService
public class OrderServiceImpl implements OrderService {

    @Autowired
    private OrderMapper orderMapper;

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

}
