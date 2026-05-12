package com.metawebthree.promotion.application.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.UpdateWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.promotion.application.service.FlashPromotionService;
import com.metawebthree.promotion.infrastructure.persistence.mapper.FlashPromotionMapper;
import com.metawebthree.promotion.infrastructure.persistence.model.FlashPromotionDO;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class FlashPromotionServiceImpl implements FlashPromotionService {

    private final FlashPromotionMapper mapper;

    @Override
    public Page<FlashPromotionDO> list(Integer pageNum, Integer pageSize, String keyword) {
        LambdaQueryWrapper<FlashPromotionDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.like(keyword != null && !keyword.isBlank(), FlashPromotionDO::getTitle, keyword);
        wrapper.orderByDesc(FlashPromotionDO::getCreateTime);
        return mapper.selectPage(new Page<>(pageNum, pageSize), wrapper);
    }

    @Override
    public void updateStatus(Long id, Integer status) {
        mapper.update(null, new UpdateWrapper<FlashPromotionDO>().eq("id", id).set("status", status));
    }

    @Override
    public void delete(Long id) {
        mapper.deleteById(id);
    }

    @Override
    public void create(FlashPromotionDO promotion) {
        mapper.insert(promotion);
    }

    @Override
    public void update(Long id, FlashPromotionDO promotion) {
        promotion.setId(id);
        mapper.updateById(promotion);
    }
}
