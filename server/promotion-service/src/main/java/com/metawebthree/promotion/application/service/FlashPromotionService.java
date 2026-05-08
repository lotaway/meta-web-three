package com.metawebthree.promotion.application.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.promotion.infrastructure.persistence.model.FlashPromotionDO;

public interface FlashPromotionService {
    Page<FlashPromotionDO> list(Integer pageNum, Integer pageSize, String keyword);

    void updateStatus(Long id, Integer status);

    void delete(Long id);

    void create(FlashPromotionDO promotion);

    void update(Long id, FlashPromotionDO promotion);
}
