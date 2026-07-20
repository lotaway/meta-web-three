package com.metawebthree.crm.domain.repository;

import com.metawebthree.crm.domain.entity.CustomerServiceTicket;

import java.util.List;
import java.util.Optional;

public interface CustomerServiceTicketRepository {
    Optional<CustomerServiceTicket> findById(Long id);
    List<CustomerServiceTicket> findAll();
    List<CustomerServiceTicket> findByStatus(String status);
    List<CustomerServiceTicket> findByPriority(String priority);
    List<CustomerServiceTicket> findByType(String type);
    List<CustomerServiceTicket> findByAssignedTo(String assignedTo);
    List<CustomerServiceTicket> findByCustomerId(Long customerId);
    List<CustomerServiceTicket> searchByKeyword(String keyword);
    CustomerServiceTicket insert(CustomerServiceTicket ticket);
    CustomerServiceTicket updateById(CustomerServiceTicket ticket);
    void deleteById(Long id);
}
