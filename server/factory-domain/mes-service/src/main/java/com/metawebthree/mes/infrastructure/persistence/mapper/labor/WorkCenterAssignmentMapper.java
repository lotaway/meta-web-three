package com.metawebthree.mes.infrastructure.persistence.mapper.labor;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.mes.infrastructure.persistence.dataobject.labor.WorkCenterAssignmentDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface WorkCenterAssignmentMapper extends BaseMapper<WorkCenterAssignmentDO> {
}
