package com.metawebthree.groupbuying.domain.repository;

import java.util.List;
import com.metawebthree.groupbuying.domain.model.GroupBuyOrderDO;

public interface GroupBuyOrderRepository {
    void save(GroupBuyOrderDO order);
    void update(GroupBuyOrderDO order);
    GroupBuyOrderDO findById(Long id);
    GroupBuyOrderDO findByOrderNo(String orderNo);
    GroupBuyOrderDO findByTeamIdAndUserId(Long teamId, Long userId);
    List<GroupBuyOrderDO> findByTeamId(Long teamId);
    List<GroupBuyOrderDO> findByUserId(Long userId);
    List<GroupBuyOrderDO> findByStatus(String status);
    void delete(Long id);
}