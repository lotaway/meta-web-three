package com.metawebthree.crm.application.command;

import com.metawebthree.crm.domain.entity.CustomerServiceTicket;
import com.metawebthree.crm.domain.exception.TicketNotFoundException;
import com.metawebthree.crm.domain.repository.CustomerServiceTicketRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class TicketCommandServiceTest {

    @Mock private CustomerServiceTicketRepository ticketRepository;

    private TicketCommandService service;

    @BeforeEach
    void setUp() {
        service = new TicketCommandService(ticketRepository);
    }

    @Test
    void create_shouldSetStatusOpenAndInsert() {
        CustomerServiceTicket ticket = new CustomerServiceTicket();
        ticket.setTitle("Test Ticket");

        CustomerServiceTicket result = service.create(ticket);

        assertEquals("OPEN", result.getStatus());
        verify(ticketRepository).insert(ticket);
    }

    @Test
    void assign_whenFoundAndOpen_shouldSetAssignedToAndStatusAssigned() {
        CustomerServiceTicket ticket = new CustomerServiceTicket();
        ticket.setId(1L);
        ticket.setStatus("OPEN");
        when(ticketRepository.findById(1L)).thenReturn(Optional.of(ticket));

        CustomerServiceTicket result = service.assign(1L, "agent@test.com");

        assertEquals("agent@test.com", result.getAssignedTo());
        assertEquals("ASSIGNED", result.getStatus());
        verify(ticketRepository).updateById(ticket);
    }

    @Test
    void assign_whenFoundAndNotOpen_shouldSetAssignedToOnly() {
        CustomerServiceTicket ticket = new CustomerServiceTicket();
        ticket.setId(1L);
        ticket.setStatus("IN_PROGRESS");
        when(ticketRepository.findById(1L)).thenReturn(Optional.of(ticket));

        CustomerServiceTicket result = service.assign(1L, "agent@test.com");

        assertEquals("agent@test.com", result.getAssignedTo());
        assertEquals("IN_PROGRESS", result.getStatus());
        verify(ticketRepository).updateById(ticket);
    }

    @Test
    void assign_whenNotFound_shouldThrow() {
        when(ticketRepository.findById(99L)).thenReturn(Optional.empty());

        assertThrows(TicketNotFoundException.class, () -> service.assign(99L, "agent"));
        verify(ticketRepository, never()).updateById(any());
    }

    @Test
    void updateStatus_shouldSetNewStatus() {
        CustomerServiceTicket ticket = new CustomerServiceTicket();
        ticket.setId(1L);
        ticket.setStatus("OPEN");
        when(ticketRepository.findById(1L)).thenReturn(Optional.of(ticket));

        CustomerServiceTicket result = service.updateStatus(1L, "IN_PROGRESS");

        assertEquals("IN_PROGRESS", result.getStatus());
        verify(ticketRepository).updateById(ticket);
    }

    @Test
    void updateStatus_whenNotFound_shouldThrow() {
        when(ticketRepository.findById(99L)).thenReturn(Optional.empty());

        assertThrows(TicketNotFoundException.class, () -> service.updateStatus(99L, "CLOSED"));
        verify(ticketRepository, never()).updateById(any());
    }

    @Test
    void delete_shouldDeleteById() {
        service.delete(1L);

        verify(ticketRepository).deleteById(1L);
    }
}
