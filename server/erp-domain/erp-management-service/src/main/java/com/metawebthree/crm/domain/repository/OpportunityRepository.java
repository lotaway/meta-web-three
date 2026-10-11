package com.metawebthree.crm.domain.repository;

import com.metawebthree.crm.domain.entity.Opportunity;

import java.util.List;
import java.util.Optional;

public interface OpportunityRepository {
    Optional<Opportunity> findById(Long id);
    List<Opportunity> findAll();
    List<Opportunity> findByStage(String stage);
    List<Opportunity> findByPipelineId(Long pipelineId);
    List<Opportunity> findByAssignedTo(String assignedTo);
    List<Opportunity> findByCustomerId(Long customerId);
    List<Opportunity> searchByKeyword(String keyword);
    Opportunity insert(Opportunity opportunity);
    Opportunity updateById(Opportunity opportunity);
    void deleteById(Long id);
}
