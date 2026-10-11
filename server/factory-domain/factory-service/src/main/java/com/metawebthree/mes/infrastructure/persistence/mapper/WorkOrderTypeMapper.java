package com.metawebthree.mes.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.mes.infrastructure.persistence.dataobject.WorkOrderTypeDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface WorkOrderTypeMapper extends BaseMapper<WorkOrderTypeDO> {
}