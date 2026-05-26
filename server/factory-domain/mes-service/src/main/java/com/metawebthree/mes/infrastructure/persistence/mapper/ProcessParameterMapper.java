package com.metawebthree.mes.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.mes.infrastructure.persistence.dataobject.ProcessParameterDO;
import org.apache.ibatis.annotations.Mapper;

/**
 * 工艺参数 Mapper
 */
@Mapper
public interface ProcessParameterMapper extends BaseMapper<ProcessParameterDO> {
}