package com.metawebthree.rma.domain.repository;

import com.metawebthree.rma.domain.entity.RmaOrderItem;
import java.util.List;

public interface RmaOrderItemRepository {
    List<RmaOrderItem> findByRmaId(Long rmaId);
    RmaOrderItem save(RmaOrderItem item);
    List<RmaOrderItem> saveAll(List<RmaOrderItem> items);
    void deleteByRmaId(Long rmaId);
}
