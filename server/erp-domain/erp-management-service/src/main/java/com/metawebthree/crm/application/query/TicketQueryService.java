package com.metawebthree.crm.application.query;

import com.metawebthree.crm.domain.entity.CustomerServiceTicket;
import com.metawebthree.crm.domain.repository.CustomerServiceTicketRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
public class TicketQueryService {

    private final CustomerServiceTicketRepository ticketRepository;

    public CustomerServiceTicket getById(Long id) {
        return ticketRepository.findById(id).orElse(null);
    }

    public List<CustomerServiceTicket> listAll() {
        return ticketRepository.findAll();
    }

    public List<CustomerServiceTicket> listByStatus(String status) {
        return ticketRepository.findByStatus(status);
    }

    public List<CustomerServiceTicket> listByPriority(String priority) {
        return ticketRepository.findByPriority(priority);
    }

    public List<CustomerServiceTicket> listByType(String type) {
        return ticketRepository.findByType(type);
    }

    public List<CustomerServiceTicket> listByAssignedTo(String assignedTo) {
        return ticketRepository.findByAssignedTo(assignedTo);
    }

    public List<CustomerServiceTicket> listByCustomerId(Long customerId) {
        return ticketRepository.findByCustomerId(customerId);
    }

    public List<CustomerServiceTicket> search(String keywords) {
        return ticketRepository.searchByKeyword(keywords);
    }
}
