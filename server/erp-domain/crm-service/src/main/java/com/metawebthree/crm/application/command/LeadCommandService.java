package com.metawebthree.crm.application.command;

import com.metawebthree.crm.domain.entity.Lead;
import com.metawebthree.crm.domain.entity.Opportunity;
import com.metawebthree.crm.domain.exception.LeadNotFoundException;
import com.metawebthree.crm.domain.repository.LeadRepository;
import com.metawebthree.crm.domain.repository.OpportunityRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
public class LeadCommandService {

    private static final String LEAD_STATUS_NEW = "NEW";
    private static final String LEAD_STATUS_CONVERTED = "CONVERTED";
    private static final String LEAD_STATUS_DISQUALIFIED = "DISQUALIFIED";

    private final LeadRepository leadRepository;
    private final OpportunityRepository opportunityRepository;

    @Transactional
    public Lead create(Lead lead) {
lead.setStatus(LEAD_STATUS_NEW);
        leadRepository.insert(lead);
        return lead;
    }

    @Transactional
    public Lead update(Lead lead) {
        Lead existing = leadRepository.findById(lead.getId()).orElse(null);
        if (existing == null) {
            throw new LeadNotFoundException(lead.getId());
        }
        leadRepository.updateById(lead);
        return lead;
    }

    @Transactional
    public void delete(Long id) {
        leadRepository.deleteById(id);
    }

    @Transactional
    public Opportunity convert(Long leadId, Opportunity opportunity) {
        Lead lead = leadRepository.findById(leadId).orElse(null);
        if (lead == null) {
            throw new LeadNotFoundException(leadId);
        }
        lead.setStatus(LEAD_STATUS_CONVERTED);
        leadRepository.updateById(lead);

        opportunity.setLeadId(leadId);
        opportunityRepository.insert(opportunity);
        return opportunity;
    }

    @Transactional
    public void disqualify(Long leadId, String reason) {
        Lead lead = leadRepository.findById(leadId).orElse(null);
        if (lead == null) {
            throw new LeadNotFoundException(leadId);
        }
        lead.setStatus(LEAD_STATUS_DISQUALIFIED);
        lead.setDescription(reason);
        leadRepository.updateById(lead);
    }
}
