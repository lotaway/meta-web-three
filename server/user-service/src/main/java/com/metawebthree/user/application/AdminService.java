package com.metawebthree.user.application;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.metawebthree.user.domain.model.AdminDO;
import com.metawebthree.user.domain.model.AdminRoleRelationDO;
import com.metawebthree.user.infrastructure.config.DefaultAdminProperties;
import com.metawebthree.user.infrastructure.config.SeedDataProperties;
import com.metawebthree.user.infrastructure.persistence.mapper.AdminMapper;
import com.metawebthree.user.infrastructure.persistence.mapper.AdminRoleRelationMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
public class AdminService extends ServiceImpl<AdminMapper, AdminDO> {

    private final AdminMapper adminMapper;
    private final AdminRoleRelationMapper adminRoleRelationMapper;
    private final PasswordEncoder passwordEncoder;
    private final DefaultAdminProperties defaultAdminProperties;
    private final SeedDataProperties seedDataProperties;

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

    public void ensureDefaultAdmin() {
        long defaultAdminId = seedDataProperties.getDefaultAdminId();
        AdminDO admin = adminMapper.selectById(defaultAdminId);
        if (admin == null) {
            admin = new AdminDO();
            admin.setId(defaultAdminId);
            admin.setUsername(defaultAdminProperties.getUsername());
            admin.setPassword(passwordEncoder.encode(defaultAdminProperties.getPassword()));
            admin.setNickName(defaultAdminProperties.getNickname());
            admin.setStatus(1);
            admin.setCreateTime(java.time.LocalDateTime.now());
            adminMapper.insert(admin);
        }
        long roleRelationCount = adminRoleRelationMapper.selectCount(
                new LambdaQueryWrapper<AdminRoleRelationDO>()
                        .eq(AdminRoleRelationDO::getAdminId, defaultAdminId)
                        .eq(AdminRoleRelationDO::getRoleId, seedDataProperties.getSuperAdminRoleId()));
        if (roleRelationCount == 0) {
            AdminRoleRelationDO rel = new AdminRoleRelationDO();
            rel.setAdminId(defaultAdminId);
            rel.setRoleId(seedDataProperties.getSuperAdminRoleId());
            adminRoleRelationMapper.insert(rel);
        }
    }

    public AdminDO login(String username, String password) {
        LambdaQueryWrapper<AdminDO> wrapper = new LambdaQueryWrapper<AdminDO>()
                .eq(AdminDO::getUsername, username);
        AdminDO admin = adminMapper.selectOne(wrapper);
        if (admin == null) {
            return null;
        }
        if (passwordEncoder.matches(password, admin.getPassword())) {
            return admin;
        }
        if (!admin.getPassword().startsWith("$2")) {
            String encoded = passwordEncoder.encode(password);
            if (password.equals(admin.getPassword())) {
                admin.setPassword(encoded);
                adminMapper.updateById(admin);
                return admin;
            }
        }
        return null;
    }

    public void changePassword(Long adminId, String oldPassword, String newPassword) {
        AdminDO admin = adminMapper.selectById(adminId);
        if (admin == null) {
            throw new IllegalArgumentException("管理员不存在");
        }
        if (!passwordEncoder.matches(oldPassword, admin.getPassword())) {
            throw new IllegalArgumentException("原密码错误");
        }
        admin.setPassword(passwordEncoder.encode(newPassword));
        adminMapper.updateById(admin);
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
