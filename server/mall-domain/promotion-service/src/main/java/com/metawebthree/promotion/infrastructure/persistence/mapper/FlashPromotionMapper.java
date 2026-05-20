package com.metawebthree.promotion.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.promotion.infrastructure.persistence.model.FlashPromotionDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface FlashPromotionMapper extends BaseMapper<FlashPromotionDO> {
}
