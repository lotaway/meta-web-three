package com.metawebthree.project.infrastructure.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.project.domain.entity.CostRecord;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface CostRecordMapper extends BaseMapper<CostRecord> {
}