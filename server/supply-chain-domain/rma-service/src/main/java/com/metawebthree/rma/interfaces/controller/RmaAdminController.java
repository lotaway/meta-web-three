package com.metawebthree.rma.interfaces.controller;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.metawebthree.common.annotations.RequirePermission;
import com.metawebthree.common.SupplyChainPermissions;
import com.metawebthree.rma.application.RmaApplicationService;
import com.metawebthree.rma.application.dto.RmaOrderDTO;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/admin/rma")
public class RmaAdminController {

    private final RmaApplicationService rmaApplicationService;

    public RmaAdminController(RmaApplicationService rmaApplicationService) {
        this.rmaApplicationService = rmaApplicationService;
    }

    @RequirePermission(SupplyChainPermissions.RMA_ADMIN)
    @GetMapping
    public IPage<RmaOrderDTO> list(@RequestParam(required = false) String status,
                                   @RequestParam(defaultValue = "1") Integer pageNum,
                                   @RequestParam(defaultValue = "1000") Integer pageSize) {
        return rmaApplicationService.listRmas(status, pageNum, pageSize);
    }

    @RequirePermission(SupplyChainPermissions.RMA_ADMIN)
    @GetMapping("/statistics")
    public Map<String, Object> statistics() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("total", rmaApplicationService.listRmas(null, 1, 1).getTotal());
        stats.put("pending", rmaApplicationService.listRmas("PENDING", 1, 1).getTotal());
        stats.put("awaitingInspection", rmaApplicationService.listRmas("AWAITING_INSPECTION", 1, 1).getTotal());
        stats.put("inspected", rmaApplicationService.listRmas("INSPECTED", 1, 1).getTotal());
        stats.put("awaitingDisposition", rmaApplicationService.listRmas("AWAITING_DISPOSITION", 1, 1).getTotal());
        stats.put("disposed", rmaApplicationService.listRmas("DISPOSED", 1, 1).getTotal());
        stats.put("completed", rmaApplicationService.listRmas("COMPLETED", 1, 1).getTotal());
        stats.put("cancelled", rmaApplicationService.listRmas("CANCELLED", 1, 1).getTotal());
        return stats;
    }

    @RequirePermission(SupplyChainPermissions.RMA_ADMIN)
    @PostMapping("/{id}/force-complete")
    public RmaOrderDTO forceComplete(@PathVariable Long id) {
        return rmaApplicationService.completeRma(id);
    }

    @RequirePermission(SupplyChainPermissions.RMA_ADMIN)
    @PostMapping("/{id}/force-cancel")
    public RmaOrderDTO forceCancel(@PathVariable Long id) {
        return rmaApplicationService.cancelRma(id);
    }
}
