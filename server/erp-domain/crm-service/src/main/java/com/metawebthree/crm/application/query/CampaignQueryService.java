package com.metawebthree.crm.application.query;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
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
        return campaignRepository.selectById(id);
    }

    public List<Campaign> listAll() {
        return campaignRepository.selectList(null);
    }

    public List<Campaign> listByStatus(String status) {
        LambdaQueryWrapper<Campaign> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(Campaign::getStatus, status);
        return campaignRepository.selectList(wrapper);
    }

    public List<Campaign> listByType(String type) {
        LambdaQueryWrapper<Campaign> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(Campaign::getType, type);
        return campaignRepository.selectList(wrapper);
    }
}
