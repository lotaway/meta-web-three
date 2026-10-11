package com.metawebthree.promotion.application.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.promotion.infrastructure.persistence.model.FlashPromotionProductRelationDO;
import java.util.List;

public interface FlashPromotionProductRelationService {
    Page<FlashPromotionProductRelationDO> list(Integer pageNum, Integer pageSize, Long flashPromotionId, Long flashPromotionSessionId);

    void create(List<FlashPromotionProductRelationDO> relations);

    void delete(Long id);

    void update(Long id, FlashPromotionProductRelationDO relation);
}
