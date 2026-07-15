package com.metawebthree.crm.application.query;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.crm.domain.entity.Opportunity;
import com.metawebthree.crm.domain.repository.OpportunityRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
@RequiredArgsConstructor
public class OpportunityQueryService {

    private final OpportunityRepository opportunityRepository;

    public Opportunity getById(Long id) {
        return opportunityRepository.selectById(id);
    }

    public List<Opportunity> listAll() {
        return opportunityRepository.selectList(null);
    }

    public List<Opportunity> listByStage(String stage) {
        LambdaQueryWrapper<Opportunity> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(Opportunity::getStage, stage);
        return opportunityRepository.selectList(wrapper);
    }

    public List<Opportunity> listByPipeline(Long pipelineId) {
        LambdaQueryWrapper<Opportunity> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(Opportunity::getPipelineId, pipelineId);
        return opportunityRepository.selectList(wrapper);
    }

    public List<Opportunity> listByAssignedTo(String assignedTo) {
        LambdaQueryWrapper<Opportunity> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(Opportunity::getAssignedTo, assignedTo);
        return opportunityRepository.selectList(wrapper);
    }

    public List<Opportunity> listByCustomerId(Long customerId) {
        LambdaQueryWrapper<Opportunity> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(Opportunity::getCustomerId, customerId);
        return opportunityRepository.selectList(wrapper);
    }

    public List<Opportunity> search(String keywords) {
        LambdaQueryWrapper<Opportunity> wrapper = new LambdaQueryWrapper<>();
        wrapper.and(w -> w.like(Opportunity::getTitle, keywords)
                          .or().like(Opportunity::getOpportunityNo, keywords)
                          .or().like(Opportunity::getCompetitor, keywords))
               .orderByAsc(Opportunity::getOpportunityNo);
        return opportunityRepository.selectList(wrapper);
    }

    public Map<String, Integer> getPipelineSummary() {
        List<Opportunity> all = listAll();
        Map<String, Integer> summary = new HashMap<>();
        for (Opportunity opp : all) {
            String stage = opp.getStage() != null ? opp.getStage() : "UNKNOWN";
            summary.merge(stage, 1, Integer::sum);
        }
        return summary;
    }
}
