package com.metawebthree.mes.infrastructure.persistence.mapper.trace;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.mes.infrastructure.persistence.dataobject.trace.TraceRecordDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface TraceRecordMapper extends BaseMapper<TraceRecordDO> {
}
