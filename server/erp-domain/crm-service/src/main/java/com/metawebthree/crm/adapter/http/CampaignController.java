package com.metawebthree.crm.adapter.http;

import com.metawebthree.crm.adapter.vo.Result;
import com.metawebthree.crm.domain.entity.Campaign;
import com.metawebthree.crm.domain.exception.CampaignNotFoundException;
import com.metawebthree.crm.domain.repository.CampaignRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/crm/campaigns")
@RequiredArgsConstructor
public class CampaignController {

    private final CampaignRepository campaignRepository;

    @GetMapping("/{id}")
    public Result<Campaign> getById(@PathVariable Long id) {
        return Result.success(campaignRepository.selectById(id));
    }

    @GetMapping("/list")
    public Result<List<Campaign>> listAll() {
        return Result.success(campaignRepository.selectList(null));
    }

    @PostMapping
    public Result<Campaign> create(@RequestBody Campaign campaign) {
        campaignRepository.insert(campaign);
        return Result.success(campaign);
    }

    @PutMapping
    public Result<Campaign> update(@RequestBody Campaign campaign) {
        Campaign existing = campaignRepository.selectById(campaign.getId());
        if (existing == null) {
            throw new CampaignNotFoundException(campaign.getId());
        }
        campaignRepository.updateById(campaign);
        return Result.success(campaign);
    }

    @DeleteMapping("/{id}")
    public Result<Void> delete(@PathVariable Long id) {
        campaignRepository.deleteById(id);
        return Result.success();
    }
}
