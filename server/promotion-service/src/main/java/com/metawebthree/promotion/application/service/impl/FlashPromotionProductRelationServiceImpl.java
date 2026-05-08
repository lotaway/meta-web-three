package com.metawebthree.promotion.application.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.promotion.application.service.FlashPromotionProductRelationService;
import com.metawebthree.promotion.infrastructure.persistence.mapper.FlashPromotionProductRelationMapper;
import com.metawebthree.promotion.infrastructure.persistence.model.FlashPromotionProductRelationDO;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import java.util.List;

@Service
@RequiredArgsConstructor
public class FlashPromotionProductRelationServiceImpl implements FlashPromotionProductRelationService {

    private final FlashPromotionProductRelationMapper mapper;

    @Override
    public Page<FlashPromotionProductRelationDO> list(Integer pageNum, Integer pageSize, Long flashPromotionId, Long flashPromotionSessionId) {
        LambdaQueryWrapper<FlashPromotionProductRelationDO> wrapper = new LambdaQueryWrapper<FlashPromotionProductRelationDO>()
                .orderByDesc(FlashPromotionProductRelationDO::getId);
        if (flashPromotionId != null) {
            wrapper.eq(FlashPromotionProductRelationDO::getFlashPromotionId, flashPromotionId);
        }
        if (flashPromotionSessionId != null) {
            wrapper.eq(FlashPromotionProductRelationDO::getFlashPromotionSessionId, flashPromotionSessionId);
        }
        return mapper.selectPage(new Page<>(pageNum, pageSize), wrapper);
    }

    @Override
    public void create(List<FlashPromotionProductRelationDO> relations) {
        for (FlashPromotionProductRelationDO r : relations) {
            mapper.insert(r);
        }
    }

    @Override
    public void delete(Long id) {
        mapper.deleteById(id);
    }

    @Override
    public void update(Long id, FlashPromotionProductRelationDO relation) {
        relation.setId(id);
        mapper.updateById(relation);
    }
}
