package com.metawebthree.user.infrastructure.persistence;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.common.annotations.PermissionChecker;
import com.metawebthree.user.domain.model.AdminRoleRelationDO;
import com.metawebthree.user.domain.model.ResourceDO;
import com.metawebthree.user.domain.model.RoleResourceRelationDO;
import com.metawebthree.user.infrastructure.persistence.mapper.AdminRoleRelationMapper;
import com.metawebthree.user.infrastructure.persistence.mapper.ResourceMapper;
import com.metawebthree.user.infrastructure.persistence.mapper.RoleResourceRelationMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

@Component
@RequiredArgsConstructor
public class AdminPermissionChecker implements PermissionChecker {

    private final AdminRoleRelationMapper adminRoleRelationMapper;
    private final RoleResourceRelationMapper roleResourceRelationMapper;
    private final ResourceMapper resourceMapper;

    @Override
    public boolean hasPermission(Long userId, String permission) {
        List<AdminRoleRelationDO> roleRelations = adminRoleRelationMapper.selectList(
                new LambdaQueryWrapper<AdminRoleRelationDO>().eq(AdminRoleRelationDO::getAdminId, userId));
        if (roleRelations.isEmpty()) {
            return true;
        }
        Set<Long> roleIds = roleRelations.stream().map(AdminRoleRelationDO::getRoleId).collect(Collectors.toSet());
        List<RoleResourceRelationDO> resourceRelations = roleResourceRelationMapper.selectList(
                new LambdaQueryWrapper<RoleResourceRelationDO>().in(RoleResourceRelationDO::getRoleId, roleIds));
        if (resourceRelations.isEmpty()) {
            return false;
        }
        Set<Long> resourceIds = resourceRelations.stream().map(RoleResourceRelationDO::getResourceId).collect(Collectors.toSet());
        if (resourceIds.isEmpty()) {
            return false;
        }
        List<ResourceDO> resources = resourceMapper.selectBatchIds(resourceIds);
        return resources.stream().anyMatch(r -> permission.equals(r.getValue()));
    }
}
