package com.metawebthree.user.interfaces.admin;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.common.annotations.RequirePermission;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.user.application.UserService;
import com.metawebthree.user.application.dto.UserDTO;
import com.metawebthree.user.domain.model.UserDO;
import com.metawebthree.common.utils.UserRole;
import com.metawebthree.user.infrastructure.persistence.mapper.UserMapper;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/admin/user")
@RequiredArgsConstructor
@Tag(name = "User Admin Controller", description = "User management for backend admin")
public class UserAdminController {

    private final UserService userService;
    private final UserMapper userMapper;

    @RequirePermission("ums:user:read")
    @Operation(summary = "Get user list")
    @GetMapping("/list")
    public ApiResponse<Page<UserDO>> list(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) String keyword,
            @RequestParam(required = false) Long typeId,
            @RequestParam(required = false) String email) {
        
        LambdaQueryWrapper<UserDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.like(keyword != null, UserDO::getEmail, keyword)
               .or(keyword != null)
               .like(keyword != null, UserDO::getNickname, keyword)
               .or()
               .like(keyword != null, UserDO::getUsername, keyword)
               .eq(email != null, UserDO::getEmail, email)
               .eq(typeId != null, UserDO::getTypeId, typeId)
               .orderByDesc(UserDO::getCreatedAt);
        
        Page<UserDO> page = new Page<>(pageNum, pageSize);
        Page<UserDO> result = userMapper.selectPage(page, wrapper);
        return ApiResponse.success(result);
    }

    @RequirePermission("ums:user:read")
    @Operation(summary = "Get user by ID")
    @GetMapping("/{id}")
    public ApiResponse<UserDO> getById(@PathVariable Long id) {
        UserDO user = userMapper.selectById(id);
        if (user == null) {
            return ApiResponse.error(ResponseStatus.NOT_FOUND, "User not found");
        }
        return ApiResponse.success(user);
    }

    @RequirePermission("ums:user:update")
    @Operation(summary = "Update user")
    @PutMapping("/{id}")
    public ApiResponse<Void> update(@PathVariable Long id, @RequestBody Map<String, Object> params) {
        UserDO user = new UserDO();
        user.setId(id);
        
        if (params.containsKey("nickname")) {
            user.setNickname((String) params.get("nickname"));
        }
        if (params.containsKey("avatar")) {
            user.setAvatar((String) params.get("avatar"));
        }
        if (params.containsKey("email")) {
            user.setEmail((String) params.get("email"));
        }
        if (params.containsKey("telephone")) {
            user.setTelephone((String) params.get("telephone"));
        }
        if (params.containsKey("username")) {
            user.setUsername((String) params.get("username"));
        }
        if (params.containsKey("integration")) {
            user.setIntegration(((Number) params.get("integration")).intValue());
        }
        if (params.containsKey("growth")) {
            user.setGrowth(((Number) params.get("growth")).intValue());
        }
        if (params.containsKey("memberLevelId")) {
            Object memberLevelId = params.get("memberLevelId");
            if (memberLevelId != null) {
                user.setMemberLevelId(((Number) memberLevelId).longValue());
            }
        }
        
        userMapper.updateById(user);
        return ApiResponse.success();
    }

    @RequirePermission("ums:user:update")
    @Operation(summary = "Update user status")
    @PutMapping("/{id}/status")
    public ApiResponse<Void> updateStatus(@PathVariable Long id, @RequestParam Integer status) {
        UserDO user = new UserDO();
        user.setId(id);
        user.setStatus(status);
        userMapper.updateById(user);
        return ApiResponse.success();
    }

    @RequirePermission("ums:user:delete")
    @Operation(summary = "Delete user")
    @DeleteMapping("/{id}")
    public ApiResponse<Void> delete(@PathVariable Long id) {
        int result = userMapper.deleteById(id);
        if (result > 0) {
            return ApiResponse.success();
        }
        return ApiResponse.error(ResponseStatus.SYSTEM_ERROR, "Failed to delete user");
    }

    @RequirePermission("ums:user:delete")
    @Operation(summary = "Batch delete users")
    @DeleteMapping("/batch")
    public ApiResponse<Void> deleteBatch(@RequestParam String ids) {
        List<Long> idList = Arrays.stream(ids.split(","))
                .filter(s -> !s.isBlank())
                .map(Long::parseLong)
                .collect(Collectors.toList());
        
        int result = userMapper.deleteBatchIds(idList);
        if (result > 0) {
            return ApiResponse.success();
        }
        return ApiResponse.error(ResponseStatus.SYSTEM_ERROR, "Failed to delete users");
    }

    @RequirePermission("ums:user:read")
    @Operation(summary = "Get user statistics")
    @GetMapping("/statistics")
    public ApiResponse<Map<String, Object>> getStatistics() {
        Map<String, Object> statistics = new HashMap<>();
        
        Long total = userMapper.selectCount(null);
        
        LambdaQueryWrapper<UserDO> activeWrapper = new LambdaQueryWrapper<>();
        activeWrapper.eq(UserDO::getStatus, 1);
        Long active = userMapper.selectCount(activeWrapper);
        
        LambdaQueryWrapper<UserDO> vipWrapper = new LambdaQueryWrapper<>();
        vipWrapper.isNotNull(UserDO::getMemberLevelId);
        Long vip = userMapper.selectCount(vipWrapper);
        
        statistics.put("totalUsers", total);
        statistics.put("activeUsers", active);
        statistics.put("vipUsers", vip);
        
        return ApiResponse.success(statistics);
    }
}