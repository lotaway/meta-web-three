package com.metawebthree.recommendation.interfaces.admin;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.recommendation.application.admin.RecommendationAdminService;
import com.metawebthree.recommendation.infrastructure.persistence.entity.RecommendationDO;
import com.metawebthree.recommendation.infrastructure.persistence.entity.RecommendationRuleDO;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api/admin/recommendation")
public class RecommendationAdminController {

    private final RecommendationAdminService adminService;

    public RecommendationAdminController(RecommendationAdminService adminService) {
        this.adminService = adminService;
    }

    // ==================== Rule Management ====================

    @GetMapping("/rule/list")
    public ApiResponse<Page<RecommendationRuleDO>> listRules(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) String ruleName,
            @RequestParam(required = false) String scene,
            @RequestParam(required = false) String status) {
        return ApiResponse.success(adminService.listRules(pageNum, pageSize, ruleName, scene, status));
    }

    @GetMapping("/rule/{id}")
    public ApiResponse<RecommendationRuleDO> getRuleById(@PathVariable Long id) {
        RecommendationRuleDO rule = adminService.getRuleById(id);
        if (rule != null) {
            return ApiResponse.success(rule);
        }
        return ApiResponse.error(ResponseStatus.NOT_FOUND, "Rule not found");
    }

    @PostMapping("/rule")
    public ApiResponse<RecommendationRuleDO> createRule(@RequestBody Map<String, Object> request) {
        return ApiResponse.success(adminService.createRule(request));
    }

    @PutMapping("/rule/{id}")
    public ApiResponse<Void> updateRule(@PathVariable Long id, @RequestBody Map<String, Object> request) {
        adminService.updateRule(id, request);
        return ApiResponse.success();
    }

    @DeleteMapping("/rule/{id}")
    public ApiResponse<Void> deleteRule(@PathVariable Long id) {
        adminService.deleteRule(id);
        return ApiResponse.success();
    }

    @PutMapping("/rule/{id}/activate")
    public ApiResponse<Void> activateRule(@PathVariable Long id) {
        adminService.activateRule(id);
        return ApiResponse.success();
    }

    @PutMapping("/rule/{id}/pause")
    public ApiResponse<Void> pauseRule(@PathVariable Long id) {
        adminService.pauseRule(id);
        return ApiResponse.success();
    }

    @PutMapping("/rule/{id}/archive")
    public ApiResponse<Void> archiveRule(@PathVariable Long id) {
        adminService.archiveRule(id);
        return ApiResponse.success();
    }

    // ==================== Recommendation Records ====================

    @GetMapping("/list")
    public ApiResponse<Page<RecommendationDO>> listRecommendations(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) Long userId,
            @RequestParam(required = false) String scene) {
        return ApiResponse.success(adminService.listRecommendations(pageNum, pageSize, userId, scene));
    }

    @GetMapping("/statistics")
    public ApiResponse<Map<String, Object>> getStatistics() {
        return ApiResponse.success(adminService.getStatistics());
    }
}