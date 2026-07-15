package com.metawebthree.crm.application.command;

import com.metawebthree.crm.domain.entity.Opportunity;
import com.metawebthree.crm.domain.exception.OpportunityNotFoundException;
import com.metawebthree.crm.domain.repository.OpportunityRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDate;
import java.util.Arrays;
import java.util.List;

@Service
@RequiredArgsConstructor
public class OpportunityCommandService {

    private final OpportunityRepository opportunityRepository;

    private static final List<String> STAGES = Arrays.asList(
            "PROSPECTING", "QUALIFICATION", "PROPOSAL", "NEGOTIATION", "CLOSED_WON", "CLOSED_LOST"
    );

    @Transactional
    public Opportunity create(Opportunity opportunity) {
        opportunityRepository.insert(opportunity);
        return opportunity;
    }

    @Transactional
    public Opportunity update(Opportunity opportunity) {
        Opportunity existing = opportunityRepository.selectById(opportunity.getId());
        if (existing == null) {
            throw new OpportunityNotFoundException(opportunity.getId());
        }
        opportunityRepository.updateById(opportunity);
        return opportunity;
    }

    @Transactional
    public void delete(Long id) {
        opportunityRepository.deleteById(id);
    }

    @Transactional
    public Opportunity advanceStage(Long id) {
        Opportunity opportunity = opportunityRepository.selectById(id);
        if (opportunity == null) {
            throw new OpportunityNotFoundException(id);
        }
        int currentIndex = STAGES.indexOf(opportunity.getStage());
        if (currentIndex >= 0 && currentIndex < STAGES.size() - 2) {
            opportunity.setStage(STAGES.get(currentIndex + 1));
            opportunityRepository.updateById(opportunity);
        }
        return opportunity;
    }

    @Transactional
    public Opportunity closeWon(Long id) {
        Opportunity opportunity = opportunityRepository.selectById(id);
        if (opportunity == null) {
            throw new OpportunityNotFoundException(id);
        }
        opportunity.setStage("CLOSED_WON");
        opportunity.setActualCloseDate(LocalDate.now());
        opportunityRepository.updateById(opportunity);
        return opportunity;
    }

    @Transactional
    public Opportunity closeLost(Long id, String reason) {
        Opportunity opportunity = opportunityRepository.selectById(id);
        if (opportunity == null) {
            throw new OpportunityNotFoundException(id);
        }
        opportunity.setStage("CLOSED_LOST");
        opportunity.setDescription(reason);
        opportunity.setActualCloseDate(LocalDate.now());
        opportunityRepository.updateById(opportunity);
        return opportunity;
    }
}
