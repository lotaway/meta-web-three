package com.metawebthree.user.application;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.metawebthree.user.domain.model.AdminDO;
import com.metawebthree.user.domain.model.AdminRoleRelationDO;
import com.metawebthree.user.infrastructure.persistence.mapper.AdminMapper;
import com.metawebthree.user.infrastructure.persistence.mapper.AdminRoleRelationMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
public class AdminService extends ServiceImpl<AdminMapper, AdminDO> {

    private final AdminMapper adminMapper;
    private final AdminRoleRelationMapper adminRoleRelationMapper;

    public Page<AdminDO> listAdmins(int pageNum, int pageSize, String keyword) {
        Page<AdminDO> page = new Page<>(pageNum, pageSize);
        LambdaQueryWrapper<AdminDO> wrapper = new LambdaQueryWrapper<>();
        if (keyword != null && !keyword.isBlank()) {
            wrapper.like(AdminDO::getUsername, keyword)
                    .or().like(AdminDO::getNickName, keyword);
        }
        wrapper.orderByDesc(AdminDO::getCreateTime);
        return adminMapper.selectPage(page, wrapper);
    }

    public AdminDO login(String username, String password) {
        LambdaQueryWrapper<AdminDO> wrapper = new LambdaQueryWrapper<AdminDO>()
                .eq(AdminDO::getUsername, username)
                .eq(AdminDO::getPassword, password);
        return adminMapper.selectOne(wrapper);
    }

    public List<AdminRoleRelationDO> getRoleRelations(Long adminId) {
        return adminRoleRelationMapper.selectList(
                new LambdaQueryWrapper<AdminRoleRelationDO>().eq(AdminRoleRelationDO::getAdminId, adminId));
    }

    public void updateRoles(Long adminId, List<Long> roleIds) {
        adminRoleRelationMapper.delete(
                new LambdaQueryWrapper<AdminRoleRelationDO>().eq(AdminRoleRelationDO::getAdminId, adminId));
        for (Long roleId : roleIds) {
            AdminRoleRelationDO rel = new AdminRoleRelationDO();
            rel.setAdminId(adminId);
            rel.setRoleId(roleId);
            adminRoleRelationMapper.insert(rel);
        }
    }
}
