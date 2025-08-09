package com.metawebthree.common.rpc.interfaces;

import java.util.Collections;
import java.util.List;

import com.metawebthree.common.dto.OrderDTO;

public interface OrderService {
    List<OrderDTO> getOrderByUserId(Long id);

    default List<OrderDTO> getOrderByUserId(String id) {
        return Collections.emptyList();
    }
}