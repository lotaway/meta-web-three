package com.metawebthree.crm.application.query;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
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
        return ticketRepository.selectById(id);
    }

    public List<CustomerServiceTicket> listAll() {
        return ticketRepository.selectList(null);
    }

    public List<CustomerServiceTicket> listByStatus(String status) {
        LambdaQueryWrapper<CustomerServiceTicket> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(CustomerServiceTicket::getStatus, status);
        return ticketRepository.selectList(wrapper);
    }

    public List<CustomerServiceTicket> listByPriority(String priority) {
        LambdaQueryWrapper<CustomerServiceTicket> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(CustomerServiceTicket::getPriority, priority);
        return ticketRepository.selectList(wrapper);
    }

    public List<CustomerServiceTicket> listByType(String type) {
        LambdaQueryWrapper<CustomerServiceTicket> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(CustomerServiceTicket::getType, type);
        return ticketRepository.selectList(wrapper);
    }

    public List<CustomerServiceTicket> listByAssignedTo(String assignedTo) {
        LambdaQueryWrapper<CustomerServiceTicket> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(CustomerServiceTicket::getAssignedTo, assignedTo);
        return ticketRepository.selectList(wrapper);
    }

    public List<CustomerServiceTicket> listByCustomerId(Long customerId) {
        LambdaQueryWrapper<CustomerServiceTicket> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(CustomerServiceTicket::getCustomerId, customerId);
        return ticketRepository.selectList(wrapper);
    }

    public List<CustomerServiceTicket> search(String keywords) {
        LambdaQueryWrapper<CustomerServiceTicket> wrapper = new LambdaQueryWrapper<>();
        wrapper.and(w -> w.like(CustomerServiceTicket::getTitle, keywords)
                          .or().like(CustomerServiceTicket::getTicketNo, keywords)
                          .or().like(CustomerServiceTicket::getDescription, keywords))
               .orderByAsc(CustomerServiceTicket::getTicketNo);
        return ticketRepository.selectList(wrapper);
    }
}
