package com.metawebthree.user.interfaces.web;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.common.annotations.RequirePermission;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.user.application.MenuService;
import com.metawebthree.user.domain.model.MenuDO;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/menu")
@RequiredArgsConstructor
@Tag(name = "Menu Controller", description = "后台菜单管理")
public class MenuController {

    private final MenuService menuService;

    @Operation(summary = "树形结构返回所有菜单")
    @GetMapping("/treeList")
    @RequirePermission("ums:menu:read")
    public ApiResponse<List<Map<String, Object>>> treeList() {
        List<MenuDO> allMenus = menuService.treeList();
        Map<Long, List<MenuDO>> parentMap = allMenus.stream()
                .collect(Collectors.groupingBy(m -> m.getParentId() != null ? m.getParentId() : 0L));
        List<Map<String, Object>> tree = buildTree(0L, parentMap);
        return ApiResponse.success(tree);
    }

    private List<Map<String, Object>> buildTree(Long parentId, Map<Long, List<MenuDO>> parentMap) {
        List<MenuDO> children = parentMap.getOrDefault(parentId, Collections.emptyList());
        List<Map<String, Object>> result = new ArrayList<>();
        for (MenuDO menu : children) {
            Map<String, Object> node = new HashMap<>();
            node.put("id", menu.getId());
            node.put("parentId", menu.getParentId());
            node.put("createTime", menu.getCreateTime());
            node.put("title", menu.getTitle());
            node.put("level", menu.getLevel());
            node.put("sort", menu.getSort());
            node.put("name", menu.getName());
            node.put("icon", menu.getIcon());
            node.put("hidden", menu.getHidden());
            node.put("children", buildTree(menu.getId(), parentMap));
            result.add(node);
        }
        return result;
    }

    @Operation(summary = "根据父ID分页查询菜单")
    @GetMapping("/list/{parentId}")
    @RequirePermission("ums:menu:read")
    public ApiResponse<List<MenuDO>> listByParent(@PathVariable Long parentId) {
        return ApiResponse.success(menuService.listByParentId(parentId));
    }

    @Operation(summary = "根据ID删除菜单")
    @PostMapping("/delete/{id}")
    @RequirePermission("ums:menu:delete")
    public ApiResponse<Void> delete(@PathVariable Long id) {
        menuService.removeById(id);
        return ApiResponse.success();
    }

    @Operation(summary = "添加菜单")
    @PostMapping("/create")
    @RequirePermission("ums:menu:create")
    public ApiResponse<Void> create(@RequestBody MenuDO menu) {
        menu.setCreateTime(LocalDateTime.now());
        menuService.save(menu);
        return ApiResponse.success();
    }

    @Operation(summary = "修改菜单")
    @PostMapping("/update/{id}")
    @RequirePermission("ums:menu:update")
    public ApiResponse<Void> update(@PathVariable Long id, @RequestBody MenuDO menu) {
        menu.setId(id);
        menuService.updateById(menu);
        return ApiResponse.success();
    }

    @Operation(summary = "根据ID获取菜单")
    @GetMapping("/{id}")
    @RequirePermission("ums:menu:read")
    public ApiResponse<MenuDO> getById(@PathVariable Long id) {
        return ApiResponse.success(menuService.getById(id));
    }

    @Operation(summary = "修改菜单显示状态")
    @PostMapping("/updateHidden/{id}")
    @RequirePermission("ums:menu:update")
    public ApiResponse<Void> updateHidden(@PathVariable Long id, @RequestParam Integer hidden) {
        MenuDO menu = new MenuDO();
        menu.setId(id);
        menu.setHidden(hidden);
        menuService.updateById(menu);
        return ApiResponse.success();
    }
}
