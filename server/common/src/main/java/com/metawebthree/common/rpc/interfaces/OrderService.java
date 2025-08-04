package com.metawebthree.common.rpc.interfaces;

import java.util.List;

import com.metawebthree.common.dto.OrderDTO;

public interface OrderService {
    List<OrderDTO> getOrderByUserId(Long id);
}