package com.metawebthree.crm.application.command;

import com.metawebthree.crm.domain.entity.Lead;
import com.metawebthree.crm.domain.entity.Opportunity;
import com.metawebthree.crm.domain.exception.LeadNotFoundException;
import com.metawebthree.crm.domain.repository.LeadRepository;
import com.metawebthree.crm.domain.repository.OpportunityRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class LeadCommandServiceTest {

    @Mock private LeadRepository leadRepository;
    @Mock private OpportunityRepository opportunityRepository;

    private LeadCommandService service;

    @BeforeEach
    void setUp() {
        service = new LeadCommandService(leadRepository, opportunityRepository);
    }

    @Test
    void create_shouldSetStatusNewAndInsert() {
        Lead lead = new Lead();
        lead.setName("Test Lead");

        Lead result = service.create(lead);

        assertEquals("NEW", result.getStatus());
        verify(leadRepository).insert(lead);
    }

    @Test
    void update_whenFound_shouldUpdate() {
        Lead lead = new Lead();
        lead.setId(1L);
        lead.setName("Updated Lead");
        when(leadRepository.findById(1L)).thenReturn(Optional.of(new Lead()));

        Lead result = service.update(lead);

        assertSame(lead, result);
        verify(leadRepository).updateById(lead);
    }

    @Test
    void update_whenNotFound_shouldThrow() {
        Lead lead = new Lead();
        lead.setId(99L);
        when(leadRepository.findById(99L)).thenReturn(Optional.empty());

        assertThrows(LeadNotFoundException.class, () -> service.update(lead));
        verify(leadRepository, never()).updateById(any());
    }

    @Test
    void delete_shouldDeleteById() {
        service.delete(1L);

        verify(leadRepository).deleteById(1L);
    }

    @Test
    void convert_whenLeadFound_shouldConvertAndInsertOpportunity() {
        Lead lead = new Lead();
        lead.setId(1L);
        when(leadRepository.findById(1L)).thenReturn(Optional.of(lead));

        Opportunity opportunity = new Opportunity();
        opportunity.setTitle("Test Opportunity");

        Opportunity result = service.convert(1L, opportunity);

        assertEquals("CONVERTED", lead.getStatus());
        assertEquals(1L, opportunity.getLeadId());
        assertSame(opportunity, result);
        verify(leadRepository).updateById(lead);
        verify(opportunityRepository).insert(opportunity);
    }

    @Test
    void convert_whenLeadNotFound_shouldThrow() {
        when(leadRepository.findById(99L)).thenReturn(Optional.empty());

        assertThrows(LeadNotFoundException.class,
                () -> service.convert(99L, new Opportunity()));
        verify(leadRepository, never()).updateById(any());
        verify(opportunityRepository, never()).insert(any());
    }

    @Test
    void disqualify_whenLeadFound_shouldSetDisqualified() {
        Lead lead = new Lead();
        lead.setId(1L);
        when(leadRepository.findById(1L)).thenReturn(Optional.of(lead));

        service.disqualify(1L, "Not interested");

        assertEquals("DISQUALIFIED", lead.getStatus());
        assertEquals("Not interested", lead.getDescription());
        verify(leadRepository).updateById(lead);
    }

    @Test
    void disqualify_whenLeadNotFound_shouldThrow() {
        when(leadRepository.findById(99L)).thenReturn(Optional.empty());

        assertThrows(LeadNotFoundException.class,
                () -> service.disqualify(99L, "Reason"));
        verify(leadRepository, never()).updateById(any());
    }
}
