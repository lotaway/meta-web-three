package com.metawebthree.crm.application.query;

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
        return opportunityRepository.findById(id).orElse(null);
    }

    public List<Opportunity> listAll() {
        return opportunityRepository.findAll();
    }

    public List<Opportunity> listByStage(String stage) {
        return opportunityRepository.findByStage(stage);
    }

    public List<Opportunity> listByPipeline(Long pipelineId) {
        return opportunityRepository.findByPipelineId(pipelineId);
    }

    public List<Opportunity> listByAssignedTo(String assignedTo) {
        return opportunityRepository.findByAssignedTo(assignedTo);
    }

    public List<Opportunity> listByCustomerId(Long customerId) {
        return opportunityRepository.findByCustomerId(customerId);
    }

    public List<Opportunity> search(String keywords) {
        return opportunityRepository.searchByKeyword(keywords);
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
