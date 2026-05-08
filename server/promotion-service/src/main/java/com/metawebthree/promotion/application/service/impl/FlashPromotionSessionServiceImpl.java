package com.metawebthree.promotion.application.service.impl;

import com.baomidou.mybatisplus.core.conditions.update.UpdateWrapper;
import com.metawebthree.promotion.application.service.FlashPromotionSessionService;
import com.metawebthree.promotion.infrastructure.persistence.mapper.FlashPromotionSessionMapper;
import com.metawebthree.promotion.infrastructure.persistence.model.FlashPromotionSessionDO;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import java.util.List;

@Service
@RequiredArgsConstructor
public class FlashPromotionSessionServiceImpl implements FlashPromotionSessionService {

    private final FlashPromotionSessionMapper mapper;

    @Override
    public List<FlashPromotionSessionDO> list() {
        return mapper.selectList(null);
    }

    @Override
    public List<FlashPromotionSessionDO> selectList(Long flashPromotionId) {
        return mapper.selectList(null);
    }

    @Override
    public void updateStatus(Long id, Integer status) {
        mapper.update(null, new UpdateWrapper<FlashPromotionSessionDO>().eq("id", id).set("status", status));
    }

    @Override
    public void delete(Long id) {
        mapper.deleteById(id);
    }

    @Override
    public void create(FlashPromotionSessionDO session) {
        mapper.insert(session);
    }

    @Override
    public void update(Long id, FlashPromotionSessionDO session) {
        session.setId(id);
        mapper.updateById(session);
    }
}
