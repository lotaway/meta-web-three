package com.metawebthree.crm.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.crm.domain.entity.Lead;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface LeadMapper extends BaseMapper<Lead> {
}
