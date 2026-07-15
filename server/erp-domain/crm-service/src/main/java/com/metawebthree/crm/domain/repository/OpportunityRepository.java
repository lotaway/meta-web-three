package com.metawebthree.crm.domain.repository;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.crm.domain.entity.Opportunity;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface OpportunityRepository extends BaseMapper<Opportunity> {
}
