package com.metawebthree.promotion.application.service.impl;

import com.baomidou.mybatisplus.core.conditions.update.UpdateWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.promotion.application.service.HomeBrandService;
import com.metawebthree.promotion.infrastructure.persistence.mapper.HomeBrandMapper;
import com.metawebthree.promotion.infrastructure.persistence.model.HomeBrandDO;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class HomeBrandServiceImpl implements HomeBrandService {

    private final HomeBrandMapper mapper;

    @Override
    public Page<HomeBrandDO> list(Integer pageNum, Integer pageSize, String brandName, Integer recommendStatus) {
        return mapper.selectPage(new Page<>(pageNum, pageSize), null);
    }

    @Override
    public void updateRecommendStatus(String ids, Integer recommendStatus) {
        List<Long> idList = Arrays.stream(ids.split(",")).map(String::trim).map(Long::parseLong).collect(Collectors.toList());
        mapper.update(null, new UpdateWrapper<HomeBrandDO>().in("id", idList).set("recommend_status", recommendStatus));
    }

    @Override
    public void delete(String ids) {
        List<Long> idList = Arrays.stream(ids.split(",")).map(String::trim).map(Long::parseLong).collect(Collectors.toList());
        mapper.deleteByIds(idList);
    }

    @Override
    public void create(List<HomeBrandDO> brands) {
        for (HomeBrandDO b : brands) mapper.insert(b);
    }

    @Override
    public void updateSort(Long id, Integer sort) {
        mapper.update(null, new UpdateWrapper<HomeBrandDO>().eq("id", id).set("sort", sort));
    }
}
