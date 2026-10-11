package com.metawebthree.crm.domain.repository;

import com.metawebthree.crm.domain.entity.Campaign;

import java.util.List;
import java.util.Optional;

public interface CampaignRepository {
    Optional<Campaign> findById(Long id);
    List<Campaign> findAll();
    List<Campaign> findByStatus(String status);
    List<Campaign> findByType(String type);
    Campaign insert(Campaign campaign);
    Campaign updateById(Campaign campaign);
    void deleteById(Long id);
}
