package com.metawebthree.socialcommerce.domain.repository;

import java.util.List;
import com.metawebthree.socialcommerce.domain.model.DistributionRewardDO;

public interface DistributionRewardRepository {
    void save(DistributionRewardDO reward);
    void update(DistributionRewardDO reward);
    DistributionRewardDO findById(Long id);
    List<DistributionRewardDO> findByReferrerId(Long referrerId);
    List<DistributionRewardDO> findByOrderId(Long orderId);
    List<DistributionRewardDO> findByStatus(String status);
    void delete(Long id);
}