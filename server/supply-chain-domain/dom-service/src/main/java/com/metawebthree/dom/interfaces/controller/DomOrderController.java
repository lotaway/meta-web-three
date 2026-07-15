package com.metawebthree.dom.interfaces.controller;

import com.metawebthree.common.annotations.RequirePermission;
import com.metawebthree.dom.application.DomApplicationService;
import com.metawebthree.dom.application.dto.*;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/dom")
public class DomOrderController {

    private final DomApplicationService domApplicationService;

    public DomOrderController(DomApplicationService domApplicationService) {
        this.domApplicationService = domApplicationService;
    }

    @RequirePermission(DomPermissions.DOM_ORDER_READ)
    @GetMapping("/{id}")
    public DomOrderDTO getById(
            @PathVariable Long id,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        return domApplicationService.getDomOrder(id);
    }

    @RequirePermission(DomPermissions.DOM_ORDER_READ)
    @GetMapping("/no/{domOrderNo}")
    public DomOrderDTO getByDomOrderNo(
            @PathVariable String domOrderNo,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        return domApplicationService.getDomOrderByNo(domOrderNo);
    }

    @RequirePermission(DomPermissions.DOM_ORDER_READ)
    @GetMapping
    public List<DomOrderDTO> list(
            @RequestParam(required = false) String status,
            @RequestParam(required = false) String domOrderNo,
            @RequestParam(defaultValue = "1") Integer pageNum,
            @RequestParam(defaultValue = "20") Integer pageSize,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        DomQueryParam param = new DomQueryParam();
        param.setStatus(status);
        param.setDomOrderNo(domOrderNo);
        param.setPageNum(pageNum);
        param.setPageSize(pageSize);
        return domApplicationService.listDomOrders(param);
    }

    @RequirePermission(DomPermissions.DOM_ORDER_CREATE)
    @PostMapping
    public DomOrderDTO create(
            @RequestBody CreateDomOrderRequest request,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        return domApplicationService.createDomOrder(request);
    }

    @RequirePermission(DomPermissions.DOM_ORDER_CREATE)
    @PostMapping("/{id}/check-atp")
    public DomOrderDTO checkAtp(
            @PathVariable Long id,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        return domApplicationService.checkAvailability(id);
    }

    @RequirePermission(DomPermissions.DOM_ORDER_CREATE)
    @PostMapping("/{id}/source")
    public DomOrderDTO source(
            @PathVariable Long id,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        return domApplicationService.sourceOrder(id);
    }

    @RequirePermission(DomPermissions.DOM_ORDER_APPROVE)
    @PostMapping("/{id}/approve")
    public FulfillmentPlanDTO approve(
            @PathVariable Long id,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        return domApplicationService.approveFulfillment(id);
    }

    @RequirePermission(DomPermissions.DOM_ORDER_CANCEL)
    @PostMapping("/{id}/cancel")
    public DomOrderDTO cancel(
            @PathVariable Long id,
            @RequestHeader(value = "X-User-Id", defaultValue = "system") String userId,
            @RequestHeader(value = "X-User-Role", defaultValue = "") String userRole) {
        return domApplicationService.cancelDomOrder(id);
    }
}
