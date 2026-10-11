package com.metawebthree.crm.application.command;

import com.metawebthree.crm.domain.entity.Opportunity;
import com.metawebthree.crm.domain.exception.OpportunityNotFoundException;
import com.metawebthree.crm.domain.repository.OpportunityRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.time.LocalDate;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class OpportunityCommandServiceTest {

    @Mock private OpportunityRepository opportunityRepository;

    private OpportunityCommandService service;

    @BeforeEach
    void setUp() {
        service = new OpportunityCommandService(opportunityRepository);
    }

    @Test
    void create_shouldInsertAndReturn() {
        Opportunity opportunity = new Opportunity();
        opportunity.setTitle("Test Opp");

        Opportunity result = service.create(opportunity);

        assertSame(opportunity, result);
        verify(opportunityRepository).insert(opportunity);
    }

    @Test
    void update_whenFound_shouldUpdate() {
        Opportunity opportunity = new Opportunity();
        opportunity.setId(1L);
        when(opportunityRepository.findById(1L)).thenReturn(Optional.of(new Opportunity()));

        Opportunity result = service.update(opportunity);

        assertSame(opportunity, result);
        verify(opportunityRepository).updateById(opportunity);
    }

    @Test
    void update_whenNotFound_shouldThrow() {
        Opportunity opportunity = new Opportunity();
        opportunity.setId(99L);
        when(opportunityRepository.findById(99L)).thenReturn(Optional.empty());

        assertThrows(OpportunityNotFoundException.class, () -> service.update(opportunity));
        verify(opportunityRepository, never()).updateById(any());
    }

    @Test
    void delete_shouldDeleteById() {
        service.delete(1L);

        verify(opportunityRepository).deleteById(1L);
    }

    @Test
    void advanceStage_fromProspecting_shouldMoveToQualification() {
        Opportunity opportunity = new Opportunity();
        opportunity.setId(1L);
        opportunity.setStage("PROSPECTING");
        when(opportunityRepository.findById(1L)).thenReturn(Optional.of(opportunity));

        Opportunity result = service.advanceStage(1L);

        assertEquals("QUALIFICATION", result.getStage());
        verify(opportunityRepository).updateById(opportunity);
    }

    @Test
    void advanceStage_whenNotFound_shouldThrow() {
        when(opportunityRepository.findById(99L)).thenReturn(Optional.empty());

        assertThrows(OpportunityNotFoundException.class, () -> service.advanceStage(99L));
        verify(opportunityRepository, never()).updateById(any());
    }

    @Test
    void closeWon_shouldSetStageAndCloseDate() {
        Opportunity opportunity = new Opportunity();
        opportunity.setId(1L);
        when(opportunityRepository.findById(1L)).thenReturn(Optional.of(opportunity));

        Opportunity result = service.closeWon(1L);

        assertEquals("CLOSED_WON", result.getStage());
        assertEquals(LocalDate.now(), result.getActualCloseDate());
        verify(opportunityRepository).updateById(opportunity);
    }

    @Test
    void closeWon_whenNotFound_shouldThrow() {
        when(opportunityRepository.findById(99L)).thenReturn(Optional.empty());

        assertThrows(OpportunityNotFoundException.class, () -> service.closeWon(99L));
        verify(opportunityRepository, never()).updateById(any());
    }

    @Test
    void closeLost_shouldSetStageReasonAndCloseDate() {
        Opportunity opportunity = new Opportunity();
        opportunity.setId(1L);
        when(opportunityRepository.findById(1L)).thenReturn(Optional.of(opportunity));

        Opportunity result = service.closeLost(1L, "Lost to competitor");

        assertEquals("CLOSED_LOST", result.getStage());
        assertEquals("Lost to competitor", result.getDescription());
        assertEquals(LocalDate.now(), result.getActualCloseDate());
        verify(opportunityRepository).updateById(opportunity);
    }
}
