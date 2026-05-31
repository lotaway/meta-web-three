package com.metawebthree.promotion.application.service.impl;

import com.baomidou.mybatisplus.core.conditions.update.UpdateWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.metawebthree.promotion.application.service.HomeBrandService;
import com.metawebthree.promotion.infrastructure.persistence.mapper.HomeBrandMapper;
import com.metawebthree.promotion.infrastructure.persistence.model.HomeBrandDO;
import org.springframework.stereotype.Service;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class HomeBrandServiceImpl extends ServiceImpl<HomeBrandMapper, HomeBrandDO> implements HomeBrandService {

    @Override
    public Page<HomeBrandDO> list(Integer pageNum, Integer pageSize, String brandName, Integer recommendStatus) {
        return baseMapper.selectPage(new Page<>(pageNum, pageSize), null);
    }

    @Override
    public void updateRecommendStatus(String ids, Integer recommendStatus) {
        List<Long> idList = Arrays.stream(ids.split(",")).map(String::trim).map(Long::parseLong).collect(Collectors.toList());
        baseMapper.update(null, new UpdateWrapper<HomeBrandDO>().in("id", idList).set("recommend_status", recommendStatus));
    }

    @Override
    public void delete(String ids) {
        List<Long> idList = Arrays.stream(ids.split(",")).map(String::trim).map(Long::parseLong).collect(Collectors.toList());
        baseMapper.deleteByIds(idList);
    }

    @Override
    public void create(List<HomeBrandDO> brands) {
        saveBatch(brands);
    }

    @Override
    public void updateSort(Long id, Integer sort) {
        baseMapper.update(null, new UpdateWrapper<HomeBrandDO>().eq("id", id).set("sort", sort));
    }
}
