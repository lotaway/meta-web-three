package com.metawebthree.rma.interfaces.controller;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.metawebthree.common.annotations.RequirePermission;
import com.metawebthree.common.SupplyChainPermissions;
import com.metawebthree.rma.application.RmaApplicationService;
import com.metawebthree.rma.application.dto.CreateRmaRequest;
import com.metawebthree.rma.application.dto.MakeDispositionRequest;
import com.metawebthree.rma.application.dto.RecordInspectionRequest;
import com.metawebthree.rma.application.dto.RmaOrderDTO;
import com.metawebthree.rma.application.dto.ReturnShippingDTO;
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
        return rmaApplicationService.listRmas(param.getStatus(), param.getRmaNo(), param.getOrderNo(),
                param.getPageNum(), param.getPageSize());
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
                                         @RequestBody RecordInspectionRequest request) {
        return rmaApplicationService.recordInspection(id, request);
    }

    @RequirePermission(SupplyChainPermissions.RMA_UPDATE)
    @PostMapping("/{id}/disposition")
    public RmaOrderDTO makeDisposition(@PathVariable Long id,
                                        @RequestBody MakeDispositionRequest request) {
        return rmaApplicationService.makeDisposition(id, request);
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

    @RequirePermission(SupplyChainPermissions.RMA_UPDATE)
    @PostMapping("/{id}/shipping")
    public ReturnShippingDTO createShipping(@PathVariable Long id,
                                             @RequestBody ReturnShippingDTO dto) {
        return rmaApplicationService.createReturnShipping(id, dto);
    }

    @RequirePermission(SupplyChainPermissions.RMA_READ)
    @GetMapping("/{id}/shipping")
    public ReturnShippingDTO getShipping(@PathVariable Long id) {
        return rmaApplicationService.getReturnShipping(id);
    }
}
