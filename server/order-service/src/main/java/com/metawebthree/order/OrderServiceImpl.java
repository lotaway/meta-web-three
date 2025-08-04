package com.metawebthree.order;

import java.math.BigDecimal;
import java.util.List;

import org.apache.dubbo.config.annotation.DubboService;

import com.metawebthree.common.dto.OrderDTO;
import com.metawebthree.common.rpc.interfaces.OrderService;

@DubboService
public class OrderServiceImpl implements OrderService {

    @Override
    public List<OrderDTO> getOrderByUserId(Long id) {
        return List.of(OrderDTO.builder().id(1L).userId(id).orderNo("1234567890").orderStatus("1").orderType("1").orderAmount(BigDecimal.valueOf(100)).orderRemark("test").build());
    }
    
}
