package com.metawebthree.order.domain.repository;

import com.metawebthree.order.domain.model.OrderReturnApply;
import org.apache.ibatis.annotations.Mapper;

import java.util.List;

@Mapper
public interface OrderReturnRepository {
    void save(OrderReturnApply apply);
    void update(OrderReturnApply apply);
    OrderReturnApply findById(Long id);
    List<OrderReturnApply> findByOrderSn(String orderSn);
    List<OrderReturnApply> findByStatus(Integer status);
    void delete(Long id);
}
