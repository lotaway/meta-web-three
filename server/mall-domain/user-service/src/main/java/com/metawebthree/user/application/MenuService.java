package com.metawebthree.user.application;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.metawebthree.user.domain.model.MenuDO;
import com.metawebthree.user.infrastructure.persistence.mapper.MenuMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
public class MenuService extends ServiceImpl<MenuMapper, MenuDO> {

    private final MenuMapper menuMapper;

    public List<MenuDO> listByParentId(Long parentId) {
        return menuMapper.selectList(
                new LambdaQueryWrapper<MenuDO>().eq(MenuDO::getParentId, parentId)
                        .orderByAsc(MenuDO::getSort));
    }

    public List<MenuDO> treeList() {
        return menuMapper.selectList(new LambdaQueryWrapper<MenuDO>().orderByAsc(MenuDO::getSort));
    }
}
