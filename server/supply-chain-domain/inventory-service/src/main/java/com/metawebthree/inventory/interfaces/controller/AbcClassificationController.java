package com.metawebthree.inventory.interfaces.controller;

import com.metawebthree.common.annotations.RequirePermission;
import com.metawebthree.inventory.application.AbcClassificationApplicationService;
import com.metawebthree.inventory.application.dto.AbcClassificationDTO;
import com.metawebthree.common.SupplyChainPermissions;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/inventory/abc")
public class AbcClassificationController {

    private final AbcClassificationApplicationService abcService;

    public AbcClassificationController(AbcClassificationApplicationService abcService) {
        this.abcService = abcService;
    }

    @RequirePermission(SupplyChainPermissions.INVENTORY_ABC_ANALYZE)
    @GetMapping("/classify")
    public List<AbcClassificationDTO> classify(
            @RequestParam(required = false) Long warehouseId,
            @RequestParam(required = false, defaultValue = "30") Integer periodDays,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        return abcService.classify(warehouseId, periodDays);
    }
}