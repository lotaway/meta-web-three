package com.metawebthree.crm.adapter.http;

import com.metawebthree.crm.adapter.vo.Result;
import com.metawebthree.crm.application.command.OpportunityCommandService;
import com.metawebthree.crm.application.query.OpportunityQueryService;
import com.metawebthree.crm.domain.entity.Opportunity;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/crm/opportunities")
@RequiredArgsConstructor
public class OpportunityController {

    private final OpportunityQueryService opportunityQueryService;
    private final OpportunityCommandService opportunityCommandService;

    @GetMapping("/{id}")
    public Result<Opportunity> getById(@PathVariable Long id) {
        return Result.success(opportunityQueryService.getById(id));
    }

    @GetMapping("/list")
    public Result<List<Opportunity>> listAll() {
        return Result.success(opportunityQueryService.listAll());
    }

    @GetMapping("/stage/{stage}")
    public Result<List<Opportunity>> listByStage(@PathVariable String stage) {
        return Result.success(opportunityQueryService.listByStage(stage));
    }

    @GetMapping("/pipeline/{pipelineId}")
    public Result<List<Opportunity>> listByPipeline(@PathVariable Long pipelineId) {
        return Result.success(opportunityQueryService.listByPipeline(pipelineId));
    }

    @GetMapping("/assigned/{assignedTo}")
    public Result<List<Opportunity>> listByAssignedTo(@PathVariable String assignedTo) {
        return Result.success(opportunityQueryService.listByAssignedTo(assignedTo));
    }

    @GetMapping("/customer/{customerId}")
    public Result<List<Opportunity>> listByCustomerId(@PathVariable Long customerId) {
        return Result.success(opportunityQueryService.listByCustomerId(customerId));
    }

    @GetMapping("/search")
    public Result<List<Opportunity>> search(@RequestParam String keywords) {
        return Result.success(opportunityQueryService.search(keywords));
    }

    @GetMapping("/summary")
    public Result<Map<String, Integer>> getPipelineSummary() {
        return Result.success(opportunityQueryService.getPipelineSummary());
    }

    @PostMapping
    public Result<Opportunity> create(@RequestBody Opportunity opportunity) {
        return Result.success(opportunityCommandService.create(opportunity));
    }

    @PutMapping
    public Result<Opportunity> update(@RequestBody Opportunity opportunity) {
        return Result.success(opportunityCommandService.update(opportunity));
    }

    @DeleteMapping("/{id}")
    public Result<Void> delete(@PathVariable Long id) {
        opportunityCommandService.delete(id);
        return Result.success();
    }

    @PostMapping("/{id}/advance")
    public Result<Opportunity> advanceStage(@PathVariable Long id) {
        return Result.success(opportunityCommandService.advanceStage(id));
    }

    @PostMapping("/{id}/close-won")
    public Result<Opportunity> closeWon(@PathVariable Long id) {
        return Result.success(opportunityCommandService.closeWon(id));
    }

    @PostMapping("/{id}/close-lost")
    public Result<Opportunity> closeLost(@PathVariable Long id, @RequestParam String reason) {
        return Result.success(opportunityCommandService.closeLost(id, reason));
    }
}
