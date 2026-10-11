package com.metawebthree.mes.infrastructure.persistence.mapper.scheduling;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.mes.infrastructure.persistence.dataobject.scheduling.ScheduleOperationDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface ScheduleOperationMapper extends BaseMapper<ScheduleOperationDO> {
}
