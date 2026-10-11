package com.metawebthree.mes.infrastructure.persistence.mapper.trace;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.mes.infrastructure.persistence.dataobject.trace.TraceModelDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface TraceModelMapper extends BaseMapper<TraceModelDO> {
}
