package com.metawebthree.groupbuying.interfaces.admin;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.groupbuying.domain.model.GroupBuyActivityDO;
import com.metawebthree.groupbuying.domain.model.GroupBuyTeamDO;
import com.metawebthree.groupbuying.domain.model.GroupBuyOrderDO;
import com.metawebthree.groupbuying.infrastructure.persistence.mapper.GroupBuyActivityMapper;
import com.metawebthree.groupbuying.infrastructure.persistence.mapper.GroupBuyTeamMapper;
import com.metawebthree.groupbuying.infrastructure.persistence.mapper.GroupBuyOrderMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.math.BigDecimal;
import java.util.Map;

@RestController
@RequestMapping("/api/admin/group-buy")
public class GroupBuyAdminController {

    @Autowired
    private GroupBuyActivityMapper activityMapper;

    @Autowired
    private GroupBuyTeamMapper teamMapper;

    @Autowired
    private GroupBuyOrderMapper orderMapper;

    @GetMapping("/activity/list")
    public ApiResponse<Page<GroupBuyActivityDO>> listActivities(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) String activityName,
            @RequestParam(required = false) Long productId,
            @RequestParam(required = false) Integer status) {

        LambdaQueryWrapper<GroupBuyActivityDO> wrapper = new LambdaQueryWrapper<GroupBuyActivityDO>()
            .like(activityName != null, GroupBuyActivityDO::getActivityName, activityName)
            .eq(productId != null, GroupBuyActivityDO::getProductId, productId)
            .eq(status != null, GroupBuyActivityDO::getStatus, status)
            .orderByDesc(GroupBuyActivityDO::getCreatedAt);

        Page<GroupBuyActivityDO> page = new Page<>(pageNum, pageSize);
        Page<GroupBuyActivityDO> result = activityMapper.selectPage(page, wrapper);

        return ApiResponse.success(result);
    }

    @GetMapping("/activity/{id}")
    public ApiResponse<GroupBuyActivityDO> getActivityById(@PathVariable Long id) {
        GroupBuyActivityDO activity = activityMapper.selectById(id);
        if (activity != null) {
            return ApiResponse.success(activity);
        }
        return ApiResponse.error(ResponseStatus.NOT_FOUND, "Activity not found");
    }

    @PostMapping("/activity")
    public ApiResponse<GroupBuyActivityDO> createActivity(@RequestBody Map<String, Object> request) {
        GroupBuyActivityDO activity = new GroupBuyActivityDO();
        activity.setActivityName((String) request.get("activityName"));
        activity.setProductId(((Number) request.get("productId")).longValue());
        activity.setProductName((String) request.get("productName"));
        activity.setSinglePrice(new BigDecimal(request.get("singlePrice").toString()));
        activity.setGroupPrice(new BigDecimal(request.get("groupPrice").toString()));
        activity.setRequiredQuantity((Integer) request.get("requiredQuantity"));
        activity.setValidityHours((Integer) request.get("validityHours"));
        activity.setCurrentQuantity(0);
        activity.setStatus((Integer) request.getOrDefault("status", 1));

        String startTime = (String) request.get("startTime");
        String endTime = (String) request.get("endTime");
        if (startTime != null) {
            activity.setStartTime(java.time.LocalDateTime.parse(startTime));
        }
        if (endTime != null) {
            activity.setEndTime(java.time.LocalDateTime.parse(endTime));
        }

        activityMapper.insert(activity);
        return ApiResponse.success(activity);
    }

    @PutMapping("/activity/{id}")
    public ApiResponse<Void> updateActivity(@PathVariable Long id, @RequestBody Map<String, Object> request) {
        GroupBuyActivityDO activity = activityMapper.selectById(id);
        if (activity == null) {
            return ApiResponse.error(ResponseStatus.NOT_FOUND, "Activity not found");
        }

        if (request.get("activityName") != null) {
            activity.setActivityName((String) request.get("activityName"));
        }
        if (request.get("status") != null) {
            activity.setStatus((Integer) request.get("status"));
        }
        if (request.get("singlePrice") != null) {
            activity.setSinglePrice(new BigDecimal(request.get("singlePrice").toString()));
        }
        if (request.get("groupPrice") != null) {
            activity.setGroupPrice(new BigDecimal(request.get("groupPrice").toString()));
        }
        if (request.get("requiredQuantity") != null) {
            activity.setRequiredQuantity((Integer) request.get("requiredQuantity"));
        }

        activityMapper.updateById(activity);
        return ApiResponse.success();
    }

    @DeleteMapping("/activity/{id}")
    public ApiResponse<Void> deleteActivity(@PathVariable Long id) {
        activityMapper.deleteById(id);
        return ApiResponse.success();
    }

    @GetMapping("/team/list")
    public ApiResponse<Page<GroupBuyTeamDO>> listTeams(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) Long activityId,
            @RequestParam(required = false) Long leaderId,
            @RequestParam(required = false) String status) {

        LambdaQueryWrapper<GroupBuyTeamDO> wrapper = new LambdaQueryWrapper<GroupBuyTeamDO>()
            .eq(activityId != null, GroupBuyTeamDO::getActivityId, activityId)
            .eq(leaderId != null, GroupBuyTeamDO::getLeaderId, leaderId)
            .eq(status != null, GroupBuyTeamDO::getStatus, status)
            .orderByDesc(GroupBuyTeamDO::getCreatedAt);

        Page<GroupBuyTeamDO> page = new Page<>(pageNum, pageSize);
        Page<GroupBuyTeamDO> result = teamMapper.selectPage(page, wrapper);

        return ApiResponse.success(result);
    }

    @GetMapping("/team/{id}")
    public ApiResponse<GroupBuyTeamDO> getTeamById(@PathVariable Long id) {
        GroupBuyTeamDO team = teamMapper.selectById(id);
        if (team != null) {
            return ApiResponse.success(team);
        }
        return ApiResponse.error(ResponseStatus.NOT_FOUND, "Team not found");
    }

    @GetMapping("/order/list")
    public ApiResponse<Page<GroupBuyOrderDO>> listOrders(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) Long teamId,
            @RequestParam(required = false) Long activityId,
            @RequestParam(required = false) Long userId,
            @RequestParam(required = false) String status) {

        LambdaQueryWrapper<GroupBuyOrderDO> wrapper = new LambdaQueryWrapper<GroupBuyOrderDO>()
            .eq(teamId != null, GroupBuyOrderDO::getTeamId, teamId)
            .eq(activityId != null, GroupBuyOrderDO::getActivityId, activityId)
            .eq(userId != null, GroupBuyOrderDO::getUserId, userId)
            .eq(status != null, GroupBuyOrderDO::getStatus, status)
            .orderByDesc(GroupBuyOrderDO::getCreatedAt);

        Page<GroupBuyOrderDO> page = new Page<>(pageNum, pageSize);
        Page<GroupBuyOrderDO> result = orderMapper.selectPage(page, wrapper);

        return ApiResponse.success(result);
    }

    @GetMapping("/statistics")
    public ApiResponse<Map<String, Object>> getStatistics() {
        Long totalActivities = activityMapper.selectCount(null);

        LambdaQueryWrapper<GroupBuyActivityDO> activeWrapper = new LambdaQueryWrapper<GroupBuyActivityDO>()
            .eq(GroupBuyActivityDO::getStatus, 1);
        Long activeActivities = activityMapper.selectCount(activeWrapper);

        Long totalTeams = teamMapper.selectCount(null);
        LambdaQueryWrapper<GroupBuyTeamDO> successWrapper = new LambdaQueryWrapper<GroupBuyTeamDO>()
            .eq(GroupBuyTeamDO::getStatus, "SUCCESS");
        Long successTeams = teamMapper.selectCount(successWrapper);

        Long totalOrders = orderMapper.selectCount(null);

        Map<String, Object> data = Map.of(
            "totalActivities", totalActivities,
            "activeActivities", activeActivities,
            "totalTeams", totalTeams,
            "successTeams", successTeams,
            "totalOrders", totalOrders
        );
        return ApiResponse.success(data);
    }
}