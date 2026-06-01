package com.metawebthree.aftersale.domain.repository;

import com.metawebthree.aftersale.domain.model.AfterSaleOrderDO;
import java.util.List;

public interface AfterSaleRepository {
    AfterSaleOrderDO save(AfterSaleOrderDO afterSaleOrder);
    AfterSaleOrderDO findById(Long id);
    List<AfterSaleOrderDO> findByUserId(Long userId);
    List<AfterSaleOrderDO> findByOrderId(Long orderId);
    List<AfterSaleOrderDO> findAll();
    boolean updateStatus(Long id, Integer status);
    boolean deleteById(Long id);
}