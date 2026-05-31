package com.metawebthree.riskcontrol.repository;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.riskcontrol.domain.RiskRule;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface RiskRuleRepository extends BaseMapper<RiskRule> {
}