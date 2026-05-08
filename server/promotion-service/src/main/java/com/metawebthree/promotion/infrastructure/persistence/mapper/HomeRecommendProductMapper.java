package com.metawebthree.promotion.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.promotion.infrastructure.persistence.model.HomeRecommendProductDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface HomeRecommendProductMapper extends BaseMapper<HomeRecommendProductDO> {
}
