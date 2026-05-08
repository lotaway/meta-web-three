package com.metawebthree.user.application;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.metawebthree.user.domain.model.RoleDO;
import com.metawebthree.user.domain.model.RoleMenuRelationDO;
import com.metawebthree.user.domain.model.RoleResourceRelationDO;
import com.metawebthree.user.infrastructure.persistence.mapper.RoleMapper;
import com.metawebthree.user.infrastructure.persistence.mapper.RoleMenuRelationMapper;
import com.metawebthree.user.infrastructure.persistence.mapper.RoleResourceRelationMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
public class AdminRoleService extends ServiceImpl<RoleMapper, RoleDO> {

    private final RoleMapper roleMapper;
    private final RoleMenuRelationMapper roleMenuRelationMapper;
    private final RoleResourceRelationMapper roleResourceRelationMapper;

    public Page<RoleDO> listRoles(int pageNum, int pageSize, String keyword) {
        Page<RoleDO> page = new Page<>(pageNum, pageSize);
        LambdaQueryWrapper<RoleDO> wrapper = new LambdaQueryWrapper<>();
        if (keyword != null && !keyword.isBlank()) {
            wrapper.like(RoleDO::getName, keyword);
        }
        wrapper.orderByDesc(RoleDO::getCreateTime);
        return roleMapper.selectPage(page, wrapper);
    }

    public List<RoleDO> listAll() {
        return roleMapper.selectList(null);
    }

    public List<RoleMenuRelationDO> getMenuRelations(Long roleId) {
        return roleMenuRelationMapper.selectList(
                new LambdaQueryWrapper<RoleMenuRelationDO>().eq(RoleMenuRelationDO::getRoleId, roleId));
    }

    public void updateMenus(Long roleId, List<Long> menuIds) {
        roleMenuRelationMapper.delete(
                new LambdaQueryWrapper<RoleMenuRelationDO>().eq(RoleMenuRelationDO::getRoleId, roleId));
        for (Long menuId : menuIds) {
            RoleMenuRelationDO rel = new RoleMenuRelationDO();
            rel.setRoleId(roleId);
            rel.setMenuId(menuId);
            roleMenuRelationMapper.insert(rel);
        }
    }

    public List<RoleResourceRelationDO> getResourceRelations(Long roleId) {
        return roleResourceRelationMapper.selectList(
                new LambdaQueryWrapper<RoleResourceRelationDO>().eq(RoleResourceRelationDO::getRoleId, roleId));
    }

    public void updateResources(Long roleId, List<Long> resourceIds) {
        roleResourceRelationMapper.delete(
                new LambdaQueryWrapper<RoleResourceRelationDO>().eq(RoleResourceRelationDO::getRoleId, roleId));
        for (Long resourceId : resourceIds) {
            RoleResourceRelationDO rel = new RoleResourceRelationDO();
            rel.setRoleId(roleId);
            rel.setResourceId(resourceId);
            roleResourceRelationMapper.insert(rel);
        }
    }
}
