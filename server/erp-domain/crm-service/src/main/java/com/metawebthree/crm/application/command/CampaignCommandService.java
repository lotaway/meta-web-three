package com.metawebthree.crm.application.command;

import com.metawebthree.crm.domain.entity.Campaign;
import com.metawebthree.crm.domain.exception.CampaignNotFoundException;
import com.metawebthree.crm.domain.repository.CampaignRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
public class CampaignCommandService {

    private final CampaignRepository campaignRepository;

    @Transactional
    public Campaign create(Campaign campaign) {
        campaignRepository.insert(campaign);
        return campaign;
    }

    @Transactional
    public Campaign update(Campaign campaign) {
        Campaign existing = campaignRepository.selectById(campaign.getId());
        if (existing == null) {
            throw new CampaignNotFoundException(campaign.getId());
        }
        campaignRepository.updateById(campaign);
        return campaign;
    }

    @Transactional
    public void delete(Long id) {
        campaignRepository.deleteById(id);
    }
}
