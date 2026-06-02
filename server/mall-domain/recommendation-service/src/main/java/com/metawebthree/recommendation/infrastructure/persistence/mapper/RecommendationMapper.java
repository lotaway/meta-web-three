package com.metawebthree.recommendation.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.recommendation.infrastructure.persistence.entity.RecommendationDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface RecommendationMapper extends BaseMapper<RecommendationDO> {
}