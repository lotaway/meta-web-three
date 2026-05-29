package com.metawebthree.logistics.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.logistics.infrastructure.persistence.dataobject.LogisticsOrderDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface LogisticsOrderMapper extends BaseMapper<LogisticsOrderDO> {
}