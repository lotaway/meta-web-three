package com.metawebthree.inventoryalert.interfaces.admin;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.inventoryalert.application.dto.InventoryAlertDTO;
import com.metawebthree.inventoryalert.application.dto.ThresholdConfigDTO;
import com.metawebthree.inventoryalert.application.service.InventoryAlertApplicationService;
import com.metawebthree.inventoryalert.domain.model.AlertLevel;
import com.metawebthree.inventoryalert.domain.model.AlertStatus;
import com.metawebthree.inventoryalert.domain.model.InventoryAlertDO;
import com.metawebthree.inventoryalert.domain.repository.InventoryAlertRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.math.BigDecimal;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/admin/inventory-alert")
public class InventoryAlertAdminController {

    @Autowired
    private InventoryAlertApplicationService alertService;

    @Autowired
    private InventoryAlertRepository alertRepository;

    @GetMapping("/list")
    public ApiResponse<Map<String, Object>> listAlerts(
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "10") Integer pageSize,
            @RequestParam(required = false) Long productId,
            @RequestParam(required = false) String productName,
            @RequestParam(required = false) Integer alertStatus,
            @RequestParam(required = false) Integer alertLevel,
            @RequestParam(required = false) String startDate,
            @RequestParam(required = false) String endDate) {

        List<InventoryAlertDO> allAlerts = alertRepository.findAll();

        if (productId != null) {
            allAlerts = allAlerts.stream()
                    .filter(a -> a.getProductId() != null && a.getProductId().equals(productId))
                    .collect(Collectors.toList());
        }
        if (productName != null && !productName.isEmpty()) {
            final String name = productName.toLowerCase();
            allAlerts = allAlerts.stream()
                    .filter(a -> a.getProductName() != null && a.getProductName().toLowerCase().contains(name))
                    .collect(Collectors.toList());
        }
        if (alertStatus != null) {
            allAlerts = allAlerts.stream()
                    .filter(a -> a.getAlertStatus() != null && a.getAlertStatus().equals(alertStatus))
                    .collect(Collectors.toList());
        }
        if (alertLevel != null) {
            allAlerts = allAlerts.stream()
                    .filter(a -> a.getAlertLevel() != null && a.getAlertLevel().equals(alertLevel))
                    .collect(Collectors.toList());
        }

        allAlerts.sort((a, b) -> {
            if (b.getAlertTime() == null) return -1;
            if (a.getAlertTime() == null) return 1;
            return b.getAlertTime().compareTo(a.getAlertTime());
        });

        int total = allAlerts.size();
        int start = (pageNum - 1) * pageSize;
        int end = Math.min(start + pageSize, total);
        List<InventoryAlertDTO> pageData = allAlerts.subList(start, end)
                .stream()
                .map(this::convertToDTO)
                .collect(Collectors.toList());

        Map<String, Object> result = new HashMap<>();
        result.put("list", pageData);
        result.put("total", total);
        result.put("pageNum", pageNum);
        result.put("pageSize", pageSize);

        return ApiResponse.success(result);
    }

    @GetMapping("/{id}")
    public ApiResponse<InventoryAlertDTO> getAlertById(@PathVariable Long id) {
        InventoryAlertDO alert = alertRepository.findById(id);
        if (alert == null) {
            return ApiResponse.error(ResponseStatus.NOT_FOUND, "Alert not found");
        }
        return ApiResponse.success(convertToDTO(alert));
    }

    @PostMapping("/resolve")
    public ApiResponse<InventoryAlertDTO> resolveAlert(@RequestParam Long id, @RequestParam(required = false) String remark) {
        try {
            InventoryAlertDTO result = alertService.processAlert(id, true, remark);
            return ApiResponse.success(result);
        } catch (Exception e) {
            return ApiResponse.error(ResponseStatus.NOT_FOUND, e.getMessage());
        }
    }

    @PostMapping("/ignore")
    public ApiResponse<InventoryAlertDTO> ignoreAlert(@RequestParam Long id, @RequestParam(required = false) String remark) {
        try {
            InventoryAlertDTO result = alertService.processAlert(id, false, remark);
            return ApiResponse.success(result);
        } catch (Exception e) {
            return ApiResponse.error(ResponseStatus.NOT_FOUND, e.getMessage());
        }
    }

    @GetMapping("/statistics")
    public ApiResponse<Map<String, Object>> getStatistics() {
        List<InventoryAlertDO> allAlerts = alertRepository.findAll();

        long total = allAlerts.size();
        long pending = allAlerts.stream()
                .filter(a -> a.getAlertStatus() != null && a.getAlertStatus().equals(AlertStatus.PENDING.getCode()))
                .count();
        long highPriority = allAlerts.stream()
                .filter(a -> a.getAlertLevel() != null && a.getAlertLevel() >= AlertLevel.HIGH.getCode())
                .filter(a -> a.getAlertStatus() != null && a.getAlertStatus() <= AlertStatus.NOTIFIED.getCode())
                .count();
        long resolved = allAlerts.stream()
                .filter(a -> a.getAlertStatus() != null && a.getAlertStatus().equals(AlertStatus.RESOLVED.getCode()))
                .count();

        Map<String, Object> stats = new HashMap<>();
        stats.put("total", total);
        stats.put("pending", pending);
        stats.put("highPriority", highPriority);
        stats.put("resolved", resolved);
        stats.put("ignored", allAlerts.stream()
                .filter(a -> a.getAlertStatus() != null && a.getAlertStatus().equals(AlertStatus.IGNORED.getCode()))
                .count());

        return ApiResponse.success(stats);
    }

    @GetMapping("/high-priority")
    public ApiResponse<List<InventoryAlertDTO>> getHighPriorityAlerts() {
        List<InventoryAlertDTO> result = alertService.getHighPriorityAlerts();
        return ApiResponse.success(result);
    }

    private InventoryAlertDTO convertToDTO(InventoryAlertDO alert) {
        InventoryAlertDTO dto = new InventoryAlertDTO();
        dto.setId(alert.getId());
        dto.setProductId(alert.getProductId());
        dto.setSkuId(alert.getSkuId());
        dto.setProductName(alert.getProductName());
        dto.setSkuCode(alert.getSkuCode());
        dto.setWarehouseId(alert.getWarehouseId());
        dto.setWarehouseName(alert.getWarehouseName());
        dto.setCurrentStock(alert.getCurrentStock());
        dto.setThreshold(alert.getThreshold());
        dto.setAlertLevel(alert.getAlertLevel());
        dto.setAlertStatus(alert.getAlertStatus());
        dto.setAlertMessage(alert.getAlertMessage());
        dto.setRemark(alert.getRemark());

        if (alert.getAlertLevel() != null) {
            for (AlertLevel level : AlertLevel.values()) {
                if (level.getCode().equals(alert.getAlertLevel())) {
                    dto.setAlertLevelDesc(level.getDesc());
                    break;
                }
            }
        }

        if (alert.getAlertStatus() != null) {
            for (AlertStatus status : AlertStatus.values()) {
                if (status.getCode().equals(alert.getAlertStatus())) {
                    dto.setAlertStatusDesc(status.getDesc());
                    break;
                }
            }
        }

        if (alert.getAlertTime() != null) {
            dto.setAlertTime(alert.getAlertTime().toString());
        }
        if (alert.getResolvedTime() != null) {
            dto.setResolvedTime(alert.getResolvedTime().toString());
        }

        return dto;
    }
}