package com.metawebthree.order;

import java.math.BigDecimal;
import java.util.List;

import org.apache.dubbo.config.annotation.DubboService;

import com.metawebthree.common.dto.OrderDTO;
import com.metawebthree.common.rpc.interfaces.OrderService;

import lombok.extern.slf4j.Slf4j;

@Slf4j
@DubboService
public class OrderServiceImpl implements OrderService {

    @Override
    public List<OrderDTO> getOrderByUserId(Long id) {
        List<OrderDTO> result = List.of(getOrderByUserIdMock(id));
        return result;
    }

    private OrderDTO getOrderByUserIdMock(Long id) {
        return new OrderDTO(id, 1L, "1234567890", "1", "1", BigDecimal.valueOf(100), "test");
    }

}
