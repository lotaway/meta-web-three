package com.metawebthree.promotion.infrastructure.persistence.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.promotion.infrastructure.persistence.model.FlashPromotionSessionDO;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface FlashPromotionSessionMapper extends BaseMapper<FlashPromotionSessionDO> {
}
