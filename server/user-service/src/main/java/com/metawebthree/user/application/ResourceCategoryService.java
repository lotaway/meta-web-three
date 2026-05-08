package com.metawebthree.user.application;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.metawebthree.user.domain.model.ResourceCategoryDO;
import com.metawebthree.user.infrastructure.persistence.mapper.ResourceCategoryMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
public class ResourceCategoryService extends ServiceImpl<ResourceCategoryMapper, ResourceCategoryDO> {

    private final ResourceCategoryMapper resourceCategoryMapper;

    public List<ResourceCategoryDO> listAll() {
        return resourceCategoryMapper.selectList(null);
    }
}
