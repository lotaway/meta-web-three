package com.metawebthree.crm.domain.repository;

import com.metawebthree.crm.domain.entity.Lead;

import java.util.List;
import java.util.Optional;

public interface LeadRepository {
    Optional<Lead> findById(Long id);
    List<Lead> findAll();
    List<Lead> findByStatus(String status);
    List<Lead> findBySource(String source);
    List<Lead> findByAssignedTo(String assignedTo);
    List<Lead> searchByKeyword(String keyword);
    Lead insert(Lead lead);
    Lead updateById(Lead lead);
    void deleteById(Long id);
}
