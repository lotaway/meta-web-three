package com.metawebthree.production.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.production.infrastructure.persistence.dataobject.ProductionScheduleDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface ProductionScheduleMapper extends BaseMapper<ProductionScheduleDO> {
}