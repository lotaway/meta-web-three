package com.metawebthree.crm.application.query;

import com.metawebthree.crm.domain.entity.Lead;
import com.metawebthree.crm.domain.repository.LeadRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
public class LeadQueryService {

    private final LeadRepository leadRepository;

    public Lead getById(Long id) {
        return leadRepository.findById(id).orElse(null);
    }

    public List<Lead> listAll() {
        return leadRepository.findAll();
    }

    public List<Lead> listByStatus(String status) {
        return leadRepository.findByStatus(status);
    }

    public List<Lead> listBySource(String source) {
        return leadRepository.findBySource(source);
    }

    public List<Lead> listByAssignedTo(String assignedTo) {
        return leadRepository.findByAssignedTo(assignedTo);
    }

    public List<Lead> search(String keywords) {
        return leadRepository.searchByKeyword(keywords);
    }
}
