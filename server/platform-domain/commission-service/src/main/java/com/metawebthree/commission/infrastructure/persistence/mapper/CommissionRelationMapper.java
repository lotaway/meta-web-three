package com.metawebthree.commission.infrastructure.persistence.mapper;

import org.apache.ibatis.annotations.Mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.commission.infrastructure.persistence.model.CommissionRelationRecord;

@Mapper
public interface CommissionRelationMapper extends BaseMapper<CommissionRelationRecord> {
}
