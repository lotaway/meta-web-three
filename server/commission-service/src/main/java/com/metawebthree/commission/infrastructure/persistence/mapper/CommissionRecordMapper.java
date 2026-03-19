package com.metawebthree.commission.infrastructure.persistence.mapper;

import org.apache.ibatis.annotations.Mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.commission.infrastructure.persistence.model.CommissionRecordRecord;

@Mapper
public interface CommissionRecordMapper extends BaseMapper<CommissionRecordRecord> {
}
