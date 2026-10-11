package com.metawebthree.groupbuying.domain.repository;

import java.util.List;
import com.metawebthree.groupbuying.domain.model.GroupBuyActivityDO;

public interface GroupBuyActivityRepository {
    void save(GroupBuyActivityDO activity);
    void update(GroupBuyActivityDO activity);
    GroupBuyActivityDO findById(Long id);
    List<GroupBuyActivityDO> findByStatus(Integer status);
    List<GroupBuyActivityDO> findActiveActivities();
    void delete(Long id);
}