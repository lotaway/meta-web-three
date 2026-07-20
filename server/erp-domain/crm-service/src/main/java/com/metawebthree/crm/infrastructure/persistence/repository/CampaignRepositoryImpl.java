package com.metawebthree.crm.infrastructure.persistence.repository;

import com.metawebthree.crm.domain.entity.Campaign;
import com.metawebthree.crm.domain.repository.CampaignRepository;
import com.metawebthree.crm.infrastructure.persistence.mapper.CampaignMapper;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public class CampaignRepositoryImpl implements CampaignRepository {

    private final CampaignMapper mapper;

    public CampaignRepositoryImpl(CampaignMapper mapper) {
        this.mapper = mapper;
    }

    @Override
    public Optional<Campaign> findById(Long id) {
        return Optional.ofNullable(mapper.selectById(id));
    }

    @Override
    public List<Campaign> findAll() {
        return mapper.selectList(null);
    }

    @Override
    public List<Campaign> findByStatus(String status) {
        return mapper.findByStatus(status);
    }

    @Override
    public List<Campaign> findByType(String type) {
        return mapper.findByType(type);
    }

    @Override
    public Campaign insert(Campaign campaign) {
        mapper.insert(campaign);
        return campaign;
    }

    @Override
    public Campaign updateById(Campaign campaign) {
        mapper.updateById(campaign);
        return campaign;
    }

    @Override
    public void deleteById(Long id) {
        mapper.deleteById(id);
    }
}
