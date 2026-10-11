package com.metawebthree.mes.infrastructure.persistence.mapper.labor;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.mes.infrastructure.persistence.dataobject.labor.TimeRecordDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface TimeRecordMapper extends BaseMapper<TimeRecordDO> {
}
