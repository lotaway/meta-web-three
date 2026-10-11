package com.metawebthree.logistics.interfaces.controller;

import com.metawebthree.logistics.application.LogisticsApplicationService;
import com.metawebthree.logistics.application.dto.LogisticsOrderDTO;
import org.springframework.web.bind.annotation.*;
import java.util.List;

/**
 * 物流管理 REST API
 * 能力: 物流跟踪、运费计算(待扩展)、配送调度
 */
@RestController
@RequestMapping("/api/logistics")
public class LogisticsController {

    private final LogisticsApplicationService logisticsService;

    public LogisticsController(LogisticsApplicationService logisticsService) {
        this.logisticsService = logisticsService;
    }

    @PostMapping("/orders")
    public LogisticsOrderDTO createOrder(@RequestBody LogisticsOrderDTO dto) {
        return logisticsService.createOrder(dto);
    }

    @GetMapping("/track/{trackingNo}")
    public LogisticsOrderDTO queryByTrackingNo(@PathVariable String trackingNo) {
        return logisticsService.queryByTrackingNo(trackingNo);
    }

    @GetMapping("/order/{orderNo}")
    public LogisticsOrderDTO queryByOrderNo(@PathVariable String orderNo) {
        return logisticsService.queryByOrderNo(orderNo);
    }

    @PutMapping("/orders/{trackingNo}/status")
    public LogisticsOrderDTO updateStatus(
            @PathVariable String trackingNo,
            @RequestParam String status) {
        return logisticsService.updateStatus(trackingNo, status);
    }

    @GetMapping("/orders")
    public List<LogisticsOrderDTO> listOrders(
            @RequestParam(required = false) Long carrierId,
            @RequestParam(required = false) String status) {
        return logisticsService.listOrders(carrierId, status);
    }
}