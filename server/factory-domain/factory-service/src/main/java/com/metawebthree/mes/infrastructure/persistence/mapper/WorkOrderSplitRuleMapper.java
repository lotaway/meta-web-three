package com.metawebthree.mes.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.mes.infrastructure.persistence.dataobject.WorkOrderSplitRuleDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface WorkOrderSplitRuleMapper extends BaseMapper<WorkOrderSplitRuleDO> {
}