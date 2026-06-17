package com.metaweb.datasource.pipeline.repository.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metaweb.datasource.pipeline.repository.entity.ColumnLineageDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface ColumnLineageMapper extends BaseMapper<ColumnLineageDO> {
}
