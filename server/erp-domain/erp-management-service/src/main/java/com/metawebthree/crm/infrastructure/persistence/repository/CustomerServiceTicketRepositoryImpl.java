package com.metawebthree.crm.infrastructure.persistence.repository;

import com.metawebthree.crm.domain.entity.CustomerServiceTicket;
import com.metawebthree.crm.domain.repository.CustomerServiceTicketRepository;
import com.metawebthree.crm.infrastructure.persistence.mapper.CustomerServiceTicketMapper;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public class CustomerServiceTicketRepositoryImpl implements CustomerServiceTicketRepository {

    private final CustomerServiceTicketMapper mapper;

    public CustomerServiceTicketRepositoryImpl(CustomerServiceTicketMapper mapper) {
        this.mapper = mapper;
    }

    @Override
    public Optional<CustomerServiceTicket> findById(Long id) {
        return Optional.ofNullable(mapper.selectById(id));
    }

    @Override
    public List<CustomerServiceTicket> findAll() {
        return mapper.selectList(null);
    }

    @Override
    public List<CustomerServiceTicket> findByStatus(String status) {
        return mapper.selectList(new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<CustomerServiceTicket>()
                .eq(CustomerServiceTicket::getStatus, status));
    }

    @Override
    public List<CustomerServiceTicket> findByPriority(String priority) {
        return mapper.selectList(new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<CustomerServiceTicket>()
                .eq(CustomerServiceTicket::getPriority, priority));
    }

    @Override
    public List<CustomerServiceTicket> findByType(String type) {
        return mapper.selectList(new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<CustomerServiceTicket>()
                .eq(CustomerServiceTicket::getType, type));
    }

    @Override
    public List<CustomerServiceTicket> findByAssignedTo(String assignedTo) {
        return mapper.selectList(new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<CustomerServiceTicket>()
                .eq(CustomerServiceTicket::getAssignedTo, assignedTo));
    }

    @Override
    public List<CustomerServiceTicket> findByCustomerId(Long customerId) {
        return mapper.selectList(new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<CustomerServiceTicket>()
                .eq(CustomerServiceTicket::getCustomerId, customerId));
    }

    @Override
    public List<CustomerServiceTicket> searchByKeyword(String keyword) {
        return mapper.selectList(new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<CustomerServiceTicket>()
                .and(w -> w.like(CustomerServiceTicket::getTitle, keyword)
                        .or().like(CustomerServiceTicket::getTicketNo, keyword)
                        .or().like(CustomerServiceTicket::getDescription, keyword))
                .orderByAsc(CustomerServiceTicket::getTicketNo));
    }

    @Override
    public CustomerServiceTicket insert(CustomerServiceTicket ticket) {
        mapper.insert(ticket);
        return ticket;
    }

    @Override
    public CustomerServiceTicket updateById(CustomerServiceTicket ticket) {
        mapper.updateById(ticket);
        return ticket;
    }

    @Override
    public void deleteById(Long id) {
        mapper.deleteById(id);
    }
}
