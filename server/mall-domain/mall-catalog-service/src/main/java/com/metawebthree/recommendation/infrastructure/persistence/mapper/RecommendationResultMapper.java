package com.metawebthree.recommendation.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.recommendation.infrastructure.persistence.entity.RecommendationResultDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface RecommendationResultMapper extends BaseMapper<RecommendationResultDO> {
}
