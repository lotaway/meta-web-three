package com.metawebthree.user.interfaces.web;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.common.annotations.RequirePermission;
import com.metawebthree.common.auth.TokenBlacklistService;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.common.utils.UserJwtUtil;
import com.metawebthree.common.utils.UserRole;
import com.metawebthree.user.application.AdminRoleService;
import com.metawebthree.user.application.AdminService;
import com.metawebthree.user.application.MenuService;
import com.metawebthree.user.domain.model.AdminDO;
import com.metawebthree.user.domain.model.AdminRoleRelationDO;
import com.metawebthree.user.domain.model.MenuDO;
import com.metawebthree.user.domain.model.RoleDO;
import com.metawebthree.user.domain.model.RoleMenuRelationDO;
import io.jsonwebtoken.Claims;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/admin")
@RequiredArgsConstructor
@Tag(name = "Admin Controller", description = "后台管理员管理")
public class AdminController {

    private final AdminService adminService;
    private final AdminRoleService adminRoleService;
    private final MenuService menuService;
    private final UserJwtUtil userJwtUtil;
    private final PasswordEncoder passwordEncoder;
    private final TokenBlacklistService tokenBlacklistService;

    @Operation(summary = "管理员登录")
    @PostMapping("/login")
    public ApiResponse<Map<String, Object>> login(@RequestBody Map<String, String> body) {
        adminService.ensureDefaultAdmin();
        String username = body.get("username");
        String password = body.get("password");
        AdminDO admin = adminService.login(username, password);
        if (admin == null) {
            return ApiResponse.error(com.metawebthree.common.enums.ResponseStatus.USER_NOT_FOUND);
        }
        String token = userJwtUtil.generate(
                String.valueOf(admin.getId()),
                userJwtUtil.generateClaimsMap(admin.getUsername(), UserRole.ADMIN));
        Map<String, Object> result = new HashMap<>();
        result.put("tokenHead", "Bearer ");
        result.put("token", token);
        return ApiResponse.success(result);
    }

    @Operation(summary = "管理员登出")
    @PostMapping("/logout")
    public ApiResponse<Void> logout(@RequestHeader("X-Original-Token") String originalToken) {
        try {
            Claims claims = userJwtUtil.tryDecode(originalToken.replace("Bearer ", "")).orElse(null);
            if (claims != null && claims.getExpiration() != null) {
                long ttl = (claims.getExpiration().getTime() - System.currentTimeMillis()) / 1000;
                if (ttl > 0) {
                    tokenBlacklistService.blacklist(originalToken, ttl);
                }
            }
        } catch (Exception e) {
            // blacklist best-effort
        }
        return ApiResponse.success();
    }

    @Operation(summary = "管理员刷新Token")
    @PostMapping("/refreshToken")
    public ApiResponse<Map<String, Object>> refreshToken(@RequestHeader("X-Original-Token") String originalToken) {
        try {
            String token = originalToken.replace("Bearer ", "");
            if (tokenBlacklistService.isBlacklisted(token)) {
                return ApiResponse.error(com.metawebthree.common.enums.ResponseStatus.USER_TOKEN_EXPIRED, "token已失效");
            }
            Claims claims = userJwtUtil.tryDecode(token).orElse(null);
            if (claims == null || userJwtUtil.isTokenExpired(claims.getExpiration())) {
                return ApiResponse.error(com.metawebthree.common.enums.ResponseStatus.USER_TOKEN_EXPIRED, "token已过期");
            }
            tokenBlacklistService.blacklist(token, Math.max(1, (claims.getExpiration().getTime() - System.currentTimeMillis()) / 1000));
            String newToken = userJwtUtil.generate(claims.getSubject(), claims);
            Map<String, Object> result = new HashMap<>();
            result.put("tokenHead", "Bearer ");
            result.put("token", newToken);
            return ApiResponse.success(result);
        } catch (Exception e) {
            return ApiResponse.error(com.metawebthree.common.enums.ResponseStatus.USER_TOKEN_EXPIRED, "token刷新失败");
        }
    }

    @Operation(summary = "管理员修改密码")
    @PostMapping("/changePassword")
    public ApiResponse<Void> changePassword(
            @RequestHeader("X-User-Id") Long userId,
            @RequestParam String oldPassword,
            @RequestParam String newPassword) {
        try {
            adminService.changePassword(userId, oldPassword, newPassword);
            return ApiResponse.success();
        } catch (IllegalArgumentException e) {
            return ApiResponse.error(com.metawebthree.common.enums.ResponseStatus.USER_PASSWORD_ERROR, e.getMessage());
        }
    }

    @Operation(summary = "获取当前管理员信息")
    @GetMapping("/info")
    public ApiResponse<Map<String, Object>> info(
            @RequestHeader("X-User-Id") Long userId) {
        AdminDO admin = adminService.getById(userId);
        if (admin == null) {
            return ApiResponse.error(com.metawebthree.common.enums.ResponseStatus.NOT_FOUND);
        }
        List<AdminRoleRelationDO> relations = adminService.getRoleRelations(userId);
        List<String> roleNames;
        List<Long> roleIds;
        if (relations.isEmpty()) {
            roleNames = Collections.singletonList("超级管理员");
            roleIds = Collections.emptyList();
        } else {
            roleIds = relations.stream().map(AdminRoleRelationDO::getRoleId).collect(Collectors.toList());
            roleNames = adminRoleService.listByIds(roleIds).stream()
                    .map(RoleDO::getName).collect(Collectors.toList());
        }
        Map<String, Object> result = new HashMap<>();
        result.put("username", admin.getUsername());
        result.put("icon", admin.getIcon() != null ? admin.getIcon() : "");
        result.put("roles", roleNames);
        result.put("menus", buildMenusByRoles(roleIds));
        return ApiResponse.success(result);
    }

    private List<Map<String, Object>> buildMenusByRoles(List<Long> roleIds) {
        Set<Long> menuIds = new HashSet<>();
        if (roleIds.isEmpty()) {
            menuIds.addAll(adminRoleService.listAll().stream()
                    .flatMap(r -> adminRoleService.getMenuRelations(r.getId()).stream())
                    .map(RoleMenuRelationDO::getMenuId)
                    .collect(Collectors.toSet()));
        } else {
            for (Long roleId : roleIds) {
                menuIds.addAll(adminRoleService.getMenuRelations(roleId).stream()
                        .map(RoleMenuRelationDO::getMenuId)
                        .collect(Collectors.toSet()));
            }
        }
        if (menuIds.isEmpty()) {
            return Collections.emptyList();
        }
        List<MenuDO> allMenus = menuService.listByIds(menuIds);
        Map<Long, List<MenuDO>> parentMap = allMenus.stream()
                .collect(Collectors.groupingBy(m -> m.getParentId() != null ? m.getParentId() : 0L));
        return buildMenuTree(0L, parentMap);
    }

    private List<Map<String, Object>> buildMenuTree(Long parentId, Map<Long, List<MenuDO>> parentMap) {
        List<MenuDO> children = parentMap.getOrDefault(parentId, Collections.emptyList());
        children.sort(Comparator.comparingInt(MenuDO::getSort));
        List<Map<String, Object>> result = new ArrayList<>();
        for (MenuDO menu : children) {
            Map<String, Object> node = new HashMap<>();
            node.put("name", menu.getName());
            node.put("title", menu.getTitle());
            node.put("icon", menu.getIcon());
            node.put("hidden", menu.getHidden());
            List<Map<String, Object>> subChildren = buildMenuTree(menu.getId(), parentMap);
            node.put("children", subChildren.isEmpty() ? null : subChildren);
            result.add(node);
        }
        return result;
    }

    @Operation(summary = "管理员注册")
    @PostMapping("/register")
    public ApiResponse<Void> register(
            @RequestHeader("X-User-Role") String userRole,
            @RequestBody AdminDO admin) {
        if (!"ADMIN".equals(userRole)) {
            return ApiResponse.error(com.metawebthree.common.enums.ResponseStatus.FORBIDDEN);
        }
        admin.setCreateTime(LocalDateTime.now());
        admin.setStatus(1);
        admin.setPassword(passwordEncoder.encode(admin.getPassword()));
        adminService.save(admin);
        return ApiResponse.success();
    }

    @RequirePermission("ums:admin:read")
    @Operation(summary = "分页获取管理员列表")
    @GetMapping("/list")
    public ApiResponse<Page<AdminDO>> list(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) String keyword) {
        return ApiResponse.success(adminService.listAdmins(pageNum, pageSize, keyword));
    }

    @RequirePermission("ums:admin:update")
    @Operation(summary = "修改管理员")
    @PostMapping("/update/{id}")
    public ApiResponse<Void> update(@PathVariable Long id, @RequestBody AdminDO admin) {
        admin.setId(id);
        adminService.updateById(admin);
        return ApiResponse.success();
    }

    @RequirePermission("ums:admin:update")
    @Operation(summary = "修改管理员状态")
    @PostMapping("/updateStatus/{id}")
    public ApiResponse<Void> updateStatus(@PathVariable Long id, @RequestParam Integer status) {
        AdminDO admin = new AdminDO();
        admin.setId(id);
        admin.setStatus(status);
        adminService.updateById(admin);
        return ApiResponse.success();
    }

    @RequirePermission("ums:admin:delete")
    @Operation(summary = "删除管理员")
    @PostMapping("/delete/{id}")
    public ApiResponse<Void> delete(@PathVariable Long id) {
        adminService.removeById(id);
        return ApiResponse.success();
    }

    @RequirePermission("ums:admin:read")
    @Operation(summary = "获取管理员角色")
    @GetMapping("/role/{adminId}")
    public ApiResponse<List<RoleDO>> getRole(@PathVariable Long adminId) {
        List<AdminRoleRelationDO> relations = adminService.getRoleRelations(adminId);
        List<Long> roleIds = relations.stream().map(AdminRoleRelationDO::getRoleId).collect(Collectors.toList());
        if (roleIds.isEmpty()) {
            return ApiResponse.success(Collections.emptyList());
        }
        List<RoleDO> roles = adminRoleService.listByIds(roleIds);
        return ApiResponse.success(roles);
    }

    @RequirePermission("ums:admin:update")
    @Operation(summary = "分配管理员角色")
    @PostMapping("/role/update")
    public ApiResponse<Void> updateRole(@RequestParam Long adminId, @RequestParam String roleIds) {
        List<Long> ids = Arrays.stream(roleIds.split(","))
                .filter(s -> !s.isBlank())
                .map(Long::parseLong)
                .collect(Collectors.toList());
        adminService.updateRoles(adminId, ids);
        return ApiResponse.success();
    }
}
