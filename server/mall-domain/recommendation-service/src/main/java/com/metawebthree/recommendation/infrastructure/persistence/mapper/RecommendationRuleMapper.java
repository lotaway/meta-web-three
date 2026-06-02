package com.metawebthree.recommendation.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.recommendation.infrastructure.persistence.entity.RecommendationRuleDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface RecommendationRuleMapper extends BaseMapper<RecommendationRuleDO> {
}