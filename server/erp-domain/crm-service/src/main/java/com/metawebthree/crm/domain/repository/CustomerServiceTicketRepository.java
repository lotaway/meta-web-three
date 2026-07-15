package com.metawebthree.crm.domain.repository;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.crm.domain.entity.CustomerServiceTicket;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface CustomerServiceTicketRepository extends BaseMapper<CustomerServiceTicket> {
}
