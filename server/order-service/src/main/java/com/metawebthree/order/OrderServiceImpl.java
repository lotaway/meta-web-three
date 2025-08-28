package com.metawebthree.order;

import java.math.BigDecimal;
import java.util.List;

import org.apache.dubbo.config.annotation.DubboService;

import com.metawebthree.common.generated.rpc.OrderService;
import com.metawebthree.common.generated.rpc.OrderServiceOuterClass.GetOrderByUserIdResponse;
import com.metawebthree.common.generated.rpc.OrderServiceOuterClass.OrderDTO;

import lombok.extern.slf4j.Slf4j;

@Slf4j
@DubboService(protocol = "tri", serialization = "protobuf")
public class OrderServiceImpl implements OrderService {

    @Override
    public GetOrderByUserIdResponse getOrderByUserId(Long id) {
        List<OrderDTO> result = List.of(getOrderByUserIdMock(id));
        return GetOrderByUserIdResponse.newBuilder().addAllOrders(result.iterator()).build();
    }

    // public List<OrderDTO> getOrderByUserId(Long id) {
    //     List<OrderDTO> result = List.of(getOrderByUserIdMock(id));
    //     return result;
    // }

    private OrderDTO getOrderByUserIdMock(Long id) {
        BigDecimal orderAmount = BigDecimal.valueOf(100);
        return OrderDTO.newBuilder().setId(id).setUserId(1234567890L).setOrderNo("1234567890").setOrderStatus("1").setOrderType("1").setOrderAmount().setRemark("test").build();
    }

}
