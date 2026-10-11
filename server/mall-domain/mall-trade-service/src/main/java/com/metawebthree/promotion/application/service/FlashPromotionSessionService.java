package com.metawebthree.promotion.application.service;

import com.metawebthree.promotion.infrastructure.persistence.model.FlashPromotionSessionDO;
import java.util.List;

public interface FlashPromotionSessionService {
    List<FlashPromotionSessionDO> list();

    List<FlashPromotionSessionDO> selectList(Long flashPromotionId);

    void updateStatus(Long id, Integer status);

    void delete(Long id);

    void create(FlashPromotionSessionDO session);

    void update(Long id, FlashPromotionSessionDO session);
}
