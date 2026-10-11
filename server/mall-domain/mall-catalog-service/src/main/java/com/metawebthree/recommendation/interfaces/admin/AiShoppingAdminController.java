package com.metawebthree.recommendation.interfaces.admin;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.recommendation.application.aishopping.AiShoppingAdminService;
import com.metawebthree.recommendation.domain.aishopping.entity.AiSearchLog;
import com.metawebthree.recommendation.domain.aishopping.entity.AiShoppingConfig;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import java.util.List;
import java.util.Map;
import org.springframework.web.bind.annotation.*;

@Tag(name = "AI Shopping Admin", description = "Backend admin endpoints for AI shopping")
@RestController
@RequestMapping("/api/admin/ai-shopping")
public class AiShoppingAdminController {

    private final AiShoppingAdminService adminService;

    public AiShoppingAdminController(AiShoppingAdminService adminService) {
        this.adminService = adminService;
    }

    // ==================== Config ====================

    @Operation(summary = "List AI shopping runtime configs")
    @GetMapping("/config")
    public ApiResponse<List<AiShoppingConfig>> listConfigs() {
        return ApiResponse.success(adminService.listConfigs());
    }

    @Operation(summary = "Save an AI shopping runtime config override")
    @PostMapping("/config")
    public ApiResponse<AiShoppingConfig> saveConfig(@RequestBody Map<String, String> request) {
        String key = request.get("configKey");
        String value = request.get("configValue");
        String description = request.get("description");
        try {
            return ApiResponse.success(adminService.saveConfig(key, value, description));
        } catch (IllegalArgumentException e) {
            return ApiResponse.error(ResponseStatus.PARAM_ERROR, e.getMessage());
        }
    }

    @Operation(summary = "Delete an AI shopping runtime config override")
    @DeleteMapping("/config/{key}")
    public ApiResponse<Void> deleteConfig(@PathVariable String key) {
        adminService.deleteConfig(key);
        return ApiResponse.success();
    }

    // ==================== Index ====================

    @Operation(summary = "Rebuild the AI shopping vector index (text/image/all)")
    @PostMapping("/index/rebuild")
    public ApiResponse<Map<String, Object>> rebuildIndex(
            @RequestParam(defaultValue = "all") String type) {
        try {
            adminService.rebuildIndex(type);
            return ApiResponse.success(Map.of("started", true, "type", type));
        } catch (IllegalStateException e) {
            return ApiResponse.error(ResponseStatus.PARAM_ERROR, e.getMessage());
        }
    }

    @Operation(summary = "Get the AI shopping index build status")
    @GetMapping("/index/status")
    public ApiResponse<Map<String, Object>> indexStatus() {
        return ApiResponse.success(adminService.indexStatus());
    }

    // ==================== Provider ====================

    @Operation(summary = "Test AI provider connectivity (embedding/image/llm)")
    @PostMapping("/provider/test")
    public ApiResponse<Map<String, Object>> testProvider(
            @RequestParam(defaultValue = "embedding") String type) {
        return ApiResponse.success(adminService.testProvider(type));
    }

    // ==================== Logs ====================

    @Operation(summary = "List recent AI shopping search logs")
    @GetMapping("/logs")
    public ApiResponse<List<AiSearchLog>> recentLogs(
            @RequestParam(defaultValue = "50") int limit) {
        return ApiResponse.success(adminService.recentLogs(limit));
    }
}
