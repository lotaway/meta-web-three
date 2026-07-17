package com.metawebthree.crm.adapter.http;

import com.metawebthree.crm.adapter.vo.Result;
import com.metawebthree.crm.application.command.CampaignCommandService;
import com.metawebthree.crm.application.query.CampaignQueryService;
import com.metawebthree.crm.domain.entity.Campaign;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/crm/campaigns")
@RequiredArgsConstructor
public class CampaignController {

    private final CampaignQueryService campaignQueryService;
    private final CampaignCommandService campaignCommandService;

    @GetMapping("/{id}")
    public Result<Campaign> getById(@PathVariable Long id) {
        return Result.success(campaignQueryService.getById(id));
    }

    @GetMapping("/list")
    public Result<List<Campaign>> listAll() {
        return Result.success(campaignQueryService.listAll());
    }

    @PostMapping
    public Result<Campaign> create(@RequestBody Campaign campaign) {
        return Result.success(campaignCommandService.create(campaign));
    }

    @PutMapping
    public Result<Campaign> update(@RequestBody Campaign campaign) {
        return Result.success(campaignCommandService.update(campaign));
    }

    @DeleteMapping("/{id}")
    public Result<Void> delete(@PathVariable Long id) {
        campaignCommandService.delete(id);
        return Result.success();
    }
}
