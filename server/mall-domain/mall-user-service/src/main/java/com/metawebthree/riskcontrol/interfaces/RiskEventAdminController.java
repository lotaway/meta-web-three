package com.metawebthree.riskcontrol.interfaces;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.riskcontrol.domain.RiskEvent;
import com.metawebthree.riskcontrol.repository.RiskEventRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api/admin/risk/event")
public class RiskEventAdminController {

    @Autowired
    private RiskEventRepository riskEventRepository;

    @GetMapping("/list")
    public ApiResponse<Page<RiskEvent>> list(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) String userId,
            @RequestParam(required = false) String scene,
            @RequestParam(required = false) String riskLevel,
            @RequestParam(required = false) String decision,
            @RequestParam(required = false) Integer status) {
        
        LambdaQueryWrapper<RiskEvent> wrapper = new LambdaQueryWrapper<RiskEvent>()
            .eq(RiskEvent::getDeleted, 0)
            .like(userId != null, RiskEvent::getUserId, userId)
            .eq(scene != null, RiskEvent::getScene, scene)
            .eq(riskLevel != null, RiskEvent::getRiskLevel, riskLevel)
            .eq(decision != null, RiskEvent::getDecision, decision)
            .eq(status != null, RiskEvent::getStatus, status)
            .orderByDesc(RiskEvent::getCreateTime);

        Page<RiskEvent> page = new Page<>(pageNum, pageSize);
        Page<RiskEvent> result = riskEventRepository.selectPage(page, wrapper);

        return ApiResponse.success(result);
    }

    @GetMapping("/{id}")
    public ApiResponse<RiskEvent> getById(@PathVariable Long id) {
        RiskEvent event = riskEventRepository.selectById(id);
        if (event != null && event.getDeleted() == 0) {
            return ApiResponse.success(event);
        }
        return ApiResponse.error(ResponseStatus.NOT_FOUND, "Risk event not found");
    }

    @PutMapping("/{id}/status")
    public ApiResponse<Void> updateStatus(@PathVariable Long id, @RequestParam Integer status) {
        RiskEvent event = new RiskEvent();
        event.setId(id);
        event.setStatus(status);
        event.setUpdateTime(System.currentTimeMillis());
        riskEventRepository.updateById(event);

        return ApiResponse.success();
    }

    @GetMapping("/statistics")
    public ApiResponse<Map<String, Object>> statistics() {
        LambdaQueryWrapper<RiskEvent> wrapper = new LambdaQueryWrapper<RiskEvent>()
            .eq(RiskEvent::getDeleted, 0);

        Long total = riskEventRepository.selectCount(wrapper);

        LambdaQueryWrapper<RiskEvent> highWrapper = new LambdaQueryWrapper<RiskEvent>()
            .eq(RiskEvent::getDeleted, 0)
            .eq(RiskEvent::getRiskLevel, "HIGH");
        Long highCount = riskEventRepository.selectCount(highWrapper);

        LambdaQueryWrapper<RiskEvent> mediumWrapper = new LambdaQueryWrapper<RiskEvent>()
            .eq(RiskEvent::getDeleted, 0)
            .eq(RiskEvent::getRiskLevel, "MEDIUM");
        Long mediumCount = riskEventRepository.selectCount(mediumWrapper);

        LambdaQueryWrapper<RiskEvent> lowWrapper = new LambdaQueryWrapper<RiskEvent>()
            .eq(RiskEvent::getDeleted, 0)
            .eq(RiskEvent::getRiskLevel, "LOW");
        Long lowCount = riskEventRepository.selectCount(lowWrapper);

        LambdaQueryWrapper<RiskEvent> reviewWrapper = new LambdaQueryWrapper<RiskEvent>()
            .eq(RiskEvent::getDeleted, 0)
            .eq(RiskEvent::getDecision, "REVIEW");
        Long reviewCount = riskEventRepository.selectCount(reviewWrapper);

        Map<String, Object> data = Map.of(
            "total", total,
            "high", highCount,
            "medium", mediumCount,
            "low", lowCount,
            "reviewPending", reviewCount
        );
        return ApiResponse.success(data);
    }
}