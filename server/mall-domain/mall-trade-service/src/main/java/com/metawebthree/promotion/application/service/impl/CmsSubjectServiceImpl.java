package com.metawebthree.promotion.application.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.promotion.application.service.CmsSubjectService;
import com.metawebthree.promotion.infrastructure.persistence.mapper.CmsSubjectMapper;
import com.metawebthree.promotion.infrastructure.persistence.model.CmsSubjectDO;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import java.util.List;

@Service
@RequiredArgsConstructor
public class CmsSubjectServiceImpl implements CmsSubjectService {
    private final CmsSubjectMapper mapper;

    @Override
    public List<CmsSubjectDO> listAll() {
        return mapper.selectList(null);
    }

    @Override
    public Page<CmsSubjectDO> list(String keyword, Integer pageNum, Integer pageSize) {
        LambdaQueryWrapper<CmsSubjectDO> wrapper = new LambdaQueryWrapper<>();
        if (keyword != null && !keyword.isBlank()) {
            wrapper.like(CmsSubjectDO::getTitle, keyword);
        }
        wrapper.orderByDesc(CmsSubjectDO::getCreateTime);
        return mapper.selectPage(new Page<>(pageNum, pageSize), wrapper);
    }

    @Override
    public CmsSubjectDO getById(Long id) {
        return mapper.selectById(id);
    }

    @Override
    public void create(CmsSubjectDO subject) {
        mapper.insert(subject);
    }

    @Override
    public void update(Long id, CmsSubjectDO subject) {
        subject.setId(id);
        mapper.updateById(subject);
    }

    @Override
    public void delete(Long id) {
        mapper.deleteById(id);
    }
}
