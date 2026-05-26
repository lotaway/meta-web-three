package com.metawebthree.mes.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.mes.infrastructure.persistence.dataobject.CodeRuleDO;
import org.apache.ibatis.annotations.Mapper;

/**
 * 编码规则 Mapper
 */
@Mapper
public interface CodeRuleMapper extends BaseMapper<CodeRuleDO> {
}