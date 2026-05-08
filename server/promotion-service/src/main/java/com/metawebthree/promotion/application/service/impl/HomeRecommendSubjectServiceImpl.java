package com.metawebthree.promotion.application.service.impl;

import com.baomidou.mybatisplus.core.conditions.update.UpdateWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.promotion.application.service.HomeRecommendSubjectService;
import com.metawebthree.promotion.infrastructure.persistence.mapper.HomeRecommendSubjectMapper;
import com.metawebthree.promotion.infrastructure.persistence.model.HomeRecommendSubjectDO;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class HomeRecommendSubjectServiceImpl implements HomeRecommendSubjectService {

    private final HomeRecommendSubjectMapper mapper;

    @Override
    public Page<HomeRecommendSubjectDO> list(Integer pageNum, Integer pageSize, String subjectName, Integer recommendStatus) {
        return mapper.selectPage(new Page<>(pageNum, pageSize), null);
    }

    @Override
    public void updateRecommendStatus(String ids, Integer recommendStatus) {
        List<Long> idList = Arrays.stream(ids.split(",")).map(String::trim).map(Long::parseLong).collect(Collectors.toList());
        mapper.update(null, new UpdateWrapper<HomeRecommendSubjectDO>().in("id", idList).set("recommend_status", recommendStatus));
    }

    @Override
    public void delete(String ids) {
        List<Long> idList = Arrays.stream(ids.split(",")).map(String::trim).map(Long::parseLong).collect(Collectors.toList());
        mapper.deleteByIds(idList);
    }

    @Override
    public void create(List<HomeRecommendSubjectDO> subjects) {
        for (HomeRecommendSubjectDO s : subjects) mapper.insert(s);
    }

    @Override
    public void updateSort(Long id, Integer sort) {
        mapper.update(null, new UpdateWrapper<HomeRecommendSubjectDO>().eq("id", id).set("sort", sort));
    }
}
