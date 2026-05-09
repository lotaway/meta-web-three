package com.metawebthree.user.interfaces.web;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.common.annotations.RequirePermission;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.user.application.AdminRoleService;
import com.metawebthree.user.domain.model.MenuDO;
import com.metawebthree.user.domain.model.ResourceDO;
import com.metawebthree.user.domain.model.RoleDO;
import com.metawebthree.user.domain.model.RoleMenuRelationDO;
import com.metawebthree.user.domain.model.RoleResourceRelationDO;
import com.metawebthree.user.application.MenuService;
import com.metawebthree.user.application.ResourceService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/role")
@RequiredArgsConstructor
@Tag(name = "Role Controller", description = "后台角色管理")
public class RoleController {

    private final AdminRoleService adminRoleService;
    private final MenuService menuService;
    private final ResourceService resourceService;

    @Operation(summary = "获取所有角色")
    @GetMapping("/listAll")
    public ApiResponse<List<RoleDO>> listAll() {
        return ApiResponse.success(adminRoleService.listAll());
    }

    @RequirePermission("ums:role:read")
    @Operation(summary = "分页获取角色列表")
    @GetMapping("/list")
    public ApiResponse<Page<RoleDO>> list(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) String keyword) {
        return ApiResponse.success(adminRoleService.listRoles(pageNum, pageSize, keyword));
    }

    @RequirePermission("ums:role:create")
    @Operation(summary = "添加角色")
    @PostMapping("/create")
    public ApiResponse<Void> create(@RequestBody RoleDO role) {
        adminRoleService.save(role);
        return ApiResponse.success();
    }

    @RequirePermission("ums:role:update")
    @Operation(summary = "修改角色")
    @PostMapping("/update/{id}")
    public ApiResponse<Void> update(@PathVariable Long id, @RequestBody RoleDO role) {
        role.setId(id);
        adminRoleService.updateById(role);
        return ApiResponse.success();
    }

    @RequirePermission("ums:role:update")
    @Operation(summary = "修改角色状态")
    @PostMapping("/updateStatus/{id}")
    public ApiResponse<Void> updateStatus(@PathVariable Long id, @RequestParam Integer status) {
        RoleDO role = new RoleDO();
        role.setId(id);
        role.setStatus(status);
        adminRoleService.updateById(role);
        return ApiResponse.success();
    }

    @RequirePermission("ums:role:delete")
    @Operation(summary = "批量删除角色")
    @PostMapping("/delete")
    public ApiResponse<Void> delete(@RequestParam String ids) {
        List<Long> idList = Arrays.stream(ids.split(","))
                .filter(s -> !s.isBlank())
                .map(Long::parseLong)
                .collect(Collectors.toList());
        adminRoleService.removeByIds(idList);
        return ApiResponse.success();
    }

    @Operation(summary = "获取角色菜单")
    @GetMapping("/listMenu/{roleId}")
    public ApiResponse<List<MenuDO>> listMenu(@PathVariable Long roleId) {
        List<RoleMenuRelationDO> relations = adminRoleService.getMenuRelations(roleId);
        List<Long> menuIds = relations.stream().map(RoleMenuRelationDO::getMenuId).collect(Collectors.toList());
        if (menuIds.isEmpty()) {
            return ApiResponse.success(Collections.emptyList());
        }
        return ApiResponse.success(menuService.listByIds(menuIds));
    }

    @RequirePermission("ums:role:update")
    @Operation(summary = "分配菜单")
    @PostMapping("/allocMenu")
    public ApiResponse<Void> allocMenu(@RequestParam Long roleId, @RequestParam String menuIds) {
        List<Long> ids = Arrays.stream(menuIds.split(","))
                .filter(s -> !s.isBlank())
                .map(Long::parseLong)
                .collect(Collectors.toList());
        adminRoleService.updateMenus(roleId, ids);
        return ApiResponse.success();
    }

    @Operation(summary = "获取角色资源")
    @GetMapping("/listResource/{roleId}")
    public ApiResponse<List<ResourceDO>> listResource(@PathVariable Long roleId) {
        List<RoleResourceRelationDO> relations = adminRoleService.getResourceRelations(roleId);
        List<Long> resourceIds = relations.stream().map(RoleResourceRelationDO::getResourceId).collect(Collectors.toList());
        if (resourceIds.isEmpty()) {
            return ApiResponse.success(Collections.emptyList());
        }
        return ApiResponse.success(resourceService.listByIds(resourceIds));
    }

    @RequirePermission("ums:role:update")
    @Operation(summary = "分配资源")
    @PostMapping("/allocResource")
    public ApiResponse<Void> allocResource(@RequestParam Long roleId, @RequestParam String resourceIds) {
        List<Long> ids = Arrays.stream(resourceIds.split(","))
                .filter(s -> !s.isBlank())
                .map(Long::parseLong)
                .collect(Collectors.toList());
        adminRoleService.updateResources(roleId, ids);
        return ApiResponse.success();
    }
}
