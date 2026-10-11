package com.metawebthree.socialcommerce.domain.repository;

import java.util.List;
import com.metawebthree.socialcommerce.domain.model.DistributionRelationDO;

public interface DistributionRelationRepository {
    void save(DistributionRelationDO relation);
    void update(DistributionRelationDO relation);
    DistributionRelationDO findById(Long id);
    DistributionRelationDO findByUserId(Long userId);
    List<DistributionRelationDO> findByReferrerId(Long referrerId);
    List<DistributionRelationDO> findByRootReferrerId(Long rootReferrerId);
    Boolean existsByUserId(Long userId);
    void delete(Long id);
}