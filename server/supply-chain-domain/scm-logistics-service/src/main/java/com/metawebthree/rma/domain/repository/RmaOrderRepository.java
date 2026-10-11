package com.metawebthree.rma.domain.repository;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.rma.domain.entity.RmaOrder;
import java.util.List;
import java.util.Optional;

public interface RmaOrderRepository {
    Optional<RmaOrder> findById(Long id);
    Optional<RmaOrder> findByRmaNo(String rmaNo);
    List<RmaOrder> findByOrderNo(String orderNo);
    List<RmaOrder> findByStatus(String status);
    List<RmaOrder> findAll();
    IPage<RmaOrder> findPage(Page<RmaOrder> page, String status);
    RmaOrder save(RmaOrder order);
}
