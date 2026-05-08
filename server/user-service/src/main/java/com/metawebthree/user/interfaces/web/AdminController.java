package com.metawebthree.user.interfaces.web;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.user.application.AdminRoleService;
import com.metawebthree.user.application.AdminService;
import com.metawebthree.user.domain.model.AdminDO;
import com.metawebthree.user.domain.model.AdminRoleRelationDO;
import com.metawebthree.user.domain.model.RoleDO;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
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

    @Operation(summary = "管理员登录")
    @PostMapping("/login")
    public ApiResponse<Map<String, Object>> login(@RequestBody Map<String, String> body) {
        String username = body.get("username");
        String password = body.get("password");
        AdminDO admin = adminService.login(username, password);
        if (admin == null) {
            return ApiResponse.error(com.metawebthree.common.enums.ResponseStatus.USER_NOT_FOUND);
        }
        Map<String, Object> result = new HashMap<>();
        result.put("tokenHead", "Bearer ");
        result.put("token", "admin-token-" + admin.getId());
        return ApiResponse.success(result);
    }

    @Operation(summary = "管理员登出")
    @PostMapping("/logout")
    public ApiResponse<Void> logout() {
        return ApiResponse.success();
    }

    @Operation(summary = "获取当前管理员信息")
    @GetMapping("/info")
    public ApiResponse<Map<String, Object>> info() {
        Map<String, Object> result = new HashMap<>();
        result.put("username", "admin");
        result.put("icon", "");
        result.put("menus", Collections.emptyList());
        result.put("roles", Collections.emptyList());
        return ApiResponse.success(result);
    }

    @Operation(summary = "管理员注册")
    @PostMapping("/register")
    public ApiResponse<Void> register(@RequestBody AdminDO admin) {
        admin.setCreateTime(LocalDateTime.now());
        admin.setStatus(1);
        adminService.save(admin);
        return ApiResponse.success();
    }

    @Operation(summary = "分页获取管理员列表")
    @GetMapping("/list")
    public ApiResponse<Page<AdminDO>> list(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) String keyword) {
        return ApiResponse.success(adminService.listAdmins(pageNum, pageSize, keyword));
    }

    @Operation(summary = "修改管理员")
    @PostMapping("/update/{id}")
    public ApiResponse<Void> update(@PathVariable Long id, @RequestBody AdminDO admin) {
        admin.setId(id);
        adminService.updateById(admin);
        return ApiResponse.success();
    }

    @Operation(summary = "修改管理员状态")
    @PostMapping("/updateStatus/{id}")
    public ApiResponse<Void> updateStatus(@PathVariable Long id, @RequestParam Integer status) {
        AdminDO admin = new AdminDO();
        admin.setId(id);
        admin.setStatus(status);
        adminService.updateById(admin);
        return ApiResponse.success();
    }

    @Operation(summary = "删除管理员")
    @PostMapping("/delete/{id}")
    public ApiResponse<Void> delete(@PathVariable Long id) {
        adminService.removeById(id);
        return ApiResponse.success();
    }

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
