package com.metawebthree.rma.interfaces.controller;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.metawebthree.common.annotations.RequirePermission;
import com.metawebthree.common.SupplyChainPermissions;
import com.metawebthree.rma.application.RmaApplicationService;
import com.metawebthree.rma.application.dto.CreateRmaRequest;
import com.metawebthree.rma.application.dto.RmaOrderDTO;
import com.metawebthree.rma.application.dto.RmaQueryParam;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/rma")
public class RmaController {

    private final RmaApplicationService rmaApplicationService;

    public RmaController(RmaApplicationService rmaApplicationService) {
        this.rmaApplicationService = rmaApplicationService;
    }

    @RequirePermission(SupplyChainPermissions.RMA_READ)
    @GetMapping("/{id}")
    public RmaOrderDTO getById(@PathVariable Long id) {
        return rmaApplicationService.getRma(id);
    }

    @RequirePermission(SupplyChainPermissions.RMA_READ)
    @GetMapping("/no/{rmaNo}")
    public RmaOrderDTO getByRmaNo(@PathVariable String rmaNo) {
        return rmaApplicationService.getRmaByNo(rmaNo);
    }

    @RequirePermission(SupplyChainPermissions.RMA_READ)
    @GetMapping
    public IPage<RmaOrderDTO> list(RmaQueryParam param) {
        return rmaApplicationService.listRmas(param.getStatus(), param.getPageNum(), param.getPageSize());
    }

    @RequirePermission(SupplyChainPermissions.RMA_CREATE)
    @PostMapping
    public RmaOrderDTO create(@RequestBody CreateRmaRequest request) {
        return rmaApplicationService.createRma(request);
    }

    @RequirePermission(SupplyChainPermissions.RMA_UPDATE)
    @PostMapping("/{id}/submit-inspection")
    public RmaOrderDTO submitInspection(@PathVariable Long id) {
        return rmaApplicationService.submitForInspection(id);
    }

    @RequirePermission(SupplyChainPermissions.RMA_UPDATE)
    @PostMapping("/{id}/record-inspection")
    public RmaOrderDTO recordInspection(@PathVariable Long id,
                                         @RequestParam String inspector,
                                         @RequestParam String result,
                                         @RequestParam(required = false) String conclusion,
                                         @RequestParam Integer totalInspected,
                                         @RequestParam Integer totalPassed,
                                         @RequestParam Integer totalFailed,
                                         @RequestParam(required = false) String remark) {
        return rmaApplicationService.recordInspection(id, inspector, result, conclusion,
                totalInspected, totalPassed, totalFailed, remark);
    }

    @RequirePermission(SupplyChainPermissions.RMA_UPDATE)
    @PostMapping("/{id}/disposition")
    public RmaOrderDTO makeDisposition(@PathVariable Long id,
                                        @RequestParam String dispositionType,
                                        @RequestParam String dispositionBy,
                                        @RequestParam(required = false) String remark) {
        return rmaApplicationService.makeDisposition(id, dispositionType, dispositionBy, remark);
    }

    @RequirePermission(SupplyChainPermissions.RMA_UPDATE)
    @PostMapping("/{id}/execute")
    public RmaOrderDTO executeDisposition(@PathVariable Long id) {
        return rmaApplicationService.executeDisposition(id);
    }

    @RequirePermission(SupplyChainPermissions.RMA_UPDATE)
    @PostMapping("/{id}/complete")
    public RmaOrderDTO complete(@PathVariable Long id) {
        return rmaApplicationService.completeRma(id);
    }

    @RequirePermission(SupplyChainPermissions.RMA_CANCEL)
    @PostMapping("/{id}/cancel")
    public RmaOrderDTO cancel(@PathVariable Long id) {
        return rmaApplicationService.cancelRma(id);
    }

    @RequirePermission(SupplyChainPermissions.RMA_READ)
    @GetMapping("/{id}/timeline")
    public List<?> timeline(@PathVariable Long id) {
        return rmaApplicationService.getRmaTimeline(id);
    }
}
