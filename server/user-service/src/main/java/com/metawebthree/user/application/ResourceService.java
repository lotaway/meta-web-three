package com.metawebthree.user.application;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.metawebthree.user.domain.model.ResourceDO;
import com.metawebthree.user.infrastructure.persistence.mapper.ResourceMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
public class ResourceService extends ServiceImpl<ResourceMapper, ResourceDO> {

    private final ResourceMapper resourceMapper;

    public Page<ResourceDO> listResources(int pageNum, int pageSize, String nameKeyword, String urlKeyword, Long categoryId) {
        Page<ResourceDO> page = new Page<>(pageNum, pageSize);
        LambdaQueryWrapper<ResourceDO> wrapper = new LambdaQueryWrapper<>();
        if (nameKeyword != null && !nameKeyword.isBlank()) {
            wrapper.like(ResourceDO::getName, nameKeyword);
        }
        if (urlKeyword != null && !urlKeyword.isBlank()) {
            wrapper.like(ResourceDO::getUrl, urlKeyword);
        }
        if (categoryId != null) {
            wrapper.eq(ResourceDO::getCategoryId, categoryId);
        }
        wrapper.orderByDesc(ResourceDO::getCreateTime);
        return resourceMapper.selectPage(page, wrapper);
    }

    public List<ResourceDO> listAll() {
        return resourceMapper.selectList(null);
    }
}
