package com.metawebthree.settlement.interfaces.controller;

import com.metawebthree.settlement.application.LogisticsSettlementApplicationService;
import com.metawebthree.settlement.domain.entity.LogisticsSettlement;
import org.springframework.web.bind.annotation.*;

import java.math.BigDecimal;
import java.util.List;
import java.util.Map;

/**
 * 物流运费结算 REST API
 * 提供物流费用结算的查询和管理接口
 */
@RestController
@RequestMapping("/api/settlement/logistics")
public class LogisticsSettlementController {
    
    private final LogisticsSettlementApplicationService settlementService;
    
    public LogisticsSettlementController(LogisticsSettlementApplicationService settlementService) {
        this.settlementService = settlementService;
    }
    
    /**
     * 根据物流追踪号查询结算单
     */
    @GetMapping("/track/{trackingNo}")
    public LogisticsSettlement getByTrackingNo(@PathVariable String trackingNo) {
        return settlementService.findByTrackingNo(trackingNo);
    }
    
    /**
     * 根据承运商查询结算单列表
     */
    @GetMapping("/carrier/{carrierId}")
    public List<LogisticsSettlement> getByCarrierId(@PathVariable Long carrierId) {
        return settlementService.findByCarrierId(carrierId);
    }
    
    /**
     * 根据状态查询结算单列表
     */
    @GetMapping("/status/{status}")
    public List<LogisticsSettlement> getByStatus(@PathVariable String status) {
        return settlementService.findByStatus(status);
    }
    
    /**
     * 查询所有结算单
     */
    @GetMapping("/list")
    public List<LogisticsSettlement> listAll() {
        return settlementService.listAllSettlements();
    }
    
    /**
     * 确认结算单
     */
    @PostMapping("/{id}/confirm")
    public Long confirm(@PathVariable Long id) {
        return settlementService.confirmSettlement(id);
    }
    
    /**
     * 处理结算（开始付款流程）
     */
    @PostMapping("/{id}/process")
    public Long process(@PathVariable Long id) {
        return settlementService.processSettlement(id);
    }
    
    /**
     * 完成结算（付款成功）
     */
    @PostMapping("/{id}/complete")
    public Long complete(@PathVariable Long id) {
        return settlementService.completeSettlement(id);
    }
    
    /**
     * 结算失败
     */
    @PostMapping("/{id}/fail")
    public Long fail(@PathVariable Long id, @RequestBody Map<String, String> request) {
        String reason = request.get("reason");
        return settlementService.failSettlement(id, reason);
    }
    
    /**
     * 取消结算单
     */
    @PostMapping("/{id}/cancel")
    public Long cancel(@PathVariable Long id) {
        return settlementService.cancelSettlement(id);
    }
    
    /**
     * 手动创建结算单（用于测试或特殊情况）
     */
    @PostMapping("/create")
    public Long create(@RequestBody Map<String, Object> request) {
        String trackingNo = (String) request.get("trackingNo");
        String orderNo = (String) request.get("orderNo");
        Long carrierId = ((Number) request.get("carrierId")).longValue();
        String carrierName = (String) request.get("carrierName");
        BigDecimal freight = new BigDecimal(request.get("freight").toString());
        
        return settlementService.autoCreateSettlement(
            trackingNo, orderNo, carrierId, carrierName, freight
        );
    }
}