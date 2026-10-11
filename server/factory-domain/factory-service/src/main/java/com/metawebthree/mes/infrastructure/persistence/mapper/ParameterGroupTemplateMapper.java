package com.metawebthree.mes.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.mes.infrastructure.persistence.dataobject.ParameterGroupTemplateDO;
import org.apache.ibatis.annotations.Mapper;

/**
 * 工艺参数组模板 Mapper
 */
@Mapper
public interface ParameterGroupTemplateMapper extends BaseMapper<ParameterGroupTemplateDO> {
}