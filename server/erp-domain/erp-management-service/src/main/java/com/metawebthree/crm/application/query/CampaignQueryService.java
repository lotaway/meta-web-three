package com.metawebthree.crm.application.query;

import com.metawebthree.crm.domain.entity.Campaign;
import com.metawebthree.crm.domain.repository.CampaignRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
public class CampaignQueryService {

    private final CampaignRepository campaignRepository;

    public Campaign getById(Long id) {
        return campaignRepository.findById(id).orElse(null);
    }

    public List<Campaign> listAll() {
        return campaignRepository.findAll();
    }

    public List<Campaign> listByStatus(String status) {
        return campaignRepository.findByStatus(status);
    }

    public List<Campaign> listByType(String type) {
        return campaignRepository.findByType(type);
    }
}
