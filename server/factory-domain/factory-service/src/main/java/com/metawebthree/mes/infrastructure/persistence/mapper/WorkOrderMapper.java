package com.metawebthree.mes.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.mes.infrastructure.persistence.dataobject.WorkOrderDO;
import org.apache.ibatis.annotations.Mapper;

/**
 * 工单 Mapper
 */
@Mapper
public interface WorkOrderMapper extends BaseMapper<WorkOrderDO> {
}