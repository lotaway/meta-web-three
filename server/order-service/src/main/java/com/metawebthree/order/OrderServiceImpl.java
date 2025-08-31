package com.metawebthree.order;

import java.util.List;
import java.util.concurrent.CompletableFuture;

import org.apache.dubbo.config.annotation.DubboService;

import com.metawebthree.common.generated.rpc.GetOrderByUserIdRequest;
import com.metawebthree.common.generated.rpc.GetOrderByUserIdResponse;
import com.metawebthree.common.generated.rpc.OrderDTO;
import com.metawebthree.common.generated.rpc.OrderService;
import com.metawebthree.common.generated.rpc.google.type.Money;

import lombok.extern.slf4j.Slf4j;

@Slf4j
@DubboService
public class OrderServiceImpl implements OrderService {

    @Override
    public GetOrderByUserIdResponse getOrderByUserId(GetOrderByUserIdRequest request) {
        Long id = request.getId();
        var result = List.of(getOrderByUserIdMock(id));
        return GetOrderByUserIdResponse.newBuilder().addAllOrders(result).build();
    }

    @Override
    public CompletableFuture<GetOrderByUserIdResponse> getOrderByUserIdAsync(GetOrderByUserIdRequest request) {
        return CompletableFuture.completedFuture(getOrderByUserId(request));
    }

    private OrderDTO getOrderByUserIdMock(Long id) {
        long orderAmount = 100L;
        var money = Money.newBuilder().setCurrencyCode("USD").setUnits(orderAmount).build();
        return OrderDTO.newBuilder().setId(id).setUserId(1234567890L).setOrderNo("1234567890").setOrderStatus("1").setOrderType("1").setOrderAmount(money).setOrderRemark("test").build();
    }

}
