package com.metawebthree.crm.application.command;

import com.metawebthree.crm.domain.entity.CustomerServiceTicket;
import com.metawebthree.crm.domain.exception.TicketNotFoundException;
import com.metawebthree.crm.domain.repository.CustomerServiceTicketRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.Arrays;
import java.util.List;

@Service
@RequiredArgsConstructor
public class TicketCommandService {

    private static final String TICKET_STATUS_OPEN = "OPEN";
    private static final String TICKET_STATUS_ASSIGNED = "ASSIGNED";

    private final CustomerServiceTicketRepository ticketRepository;

    private static final List<String> STATUS_FLOW = Arrays.asList(
            "OPEN", "ASSIGNED", "IN_PROGRESS", "RESOLVED", "CLOSED"
    );

    @Transactional
    public CustomerServiceTicket create(CustomerServiceTicket ticket) {
        ticket.setStatus(TICKET_STATUS_OPEN);
        ticketRepository.insert(ticket);
        return ticket;
    }

    @Transactional
    public CustomerServiceTicket assign(Long id, String assignedTo) {
        CustomerServiceTicket ticket = ticketRepository.findById(id).orElse(null);
        if (ticket == null) {
            throw new TicketNotFoundException(id);
        }
        ticket.setAssignedTo(assignedTo);
        if (TICKET_STATUS_OPEN.equals(ticket.getStatus())) {
            ticket.setStatus(TICKET_STATUS_ASSIGNED);
        }
        ticketRepository.updateById(ticket);
        return ticket;
    }

    @Transactional
    public CustomerServiceTicket updateStatus(Long id, String newStatus) {
        CustomerServiceTicket ticket = ticketRepository.findById(id).orElse(null);
        if (ticket == null) {
            throw new TicketNotFoundException(id);
        }
        ticket.setStatus(newStatus);
        ticketRepository.updateById(ticket);
        return ticket;
    }

    @Transactional
    public void delete(Long id) {
        ticketRepository.deleteById(id);
    }
}
