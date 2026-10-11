package com.metawebthree.settlement.application;

import com.metawebthree.settlement.domain.entity.LogisticsSettlement;
import com.metawebthree.settlement.domain.service.LogisticsSettlementDomainService;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.util.List;

/**
 * 物流费用结算应用服务
 * 处理物流运费自动结算的业务流程
 */
@Service
public class LogisticsSettlementApplicationService {
    
    private final LogisticsSettlementDomainService domainService;
    
    public LogisticsSettlementApplicationService(LogisticsSettlementDomainService domainService) {
        this.domainService = domainService;
    }
    
    /**
     * 自动创建物流运费结算单（当物流订单送达时由事件触发）
     * @param trackingNo 物流追踪号
     * @param orderNo 订单号
     * @param carrierId 承运商ID
     * @param carrierName 承运商名称
     * @param freight 运费
     * @return 结算单ID
     */
    public Long autoCreateSettlement(String trackingNo, String orderNo,
                                      Long carrierId, String carrierName,
                                      BigDecimal freight) {
        LogisticsSettlement settlement = domainService.createSettlement(
            trackingNo, orderNo, carrierId, carrierName, freight
        );
        return settlement.getId();
    }
    
    /**
     * 确认结算单
     */
    public Long confirmSettlement(Long id) {
        LogisticsSettlement settlement = domainService.confirmSettlement(id);
        return settlement.getId();
    }
    
    /**
     * 处理结算
     */
    public Long processSettlement(Long id) {
        LogisticsSettlement settlement = domainService.processSettlement(id);
        return settlement.getId();
    }
    
    /**
     * 完成结算（付款成功）
     */
    public Long completeSettlement(Long id) {
        LogisticsSettlement settlement = domainService.completeSettlement(id);
        return settlement.getId();
    }
    
    /**
     * 结算失败
     */
    public Long failSettlement(Long id, String reason) {
        LogisticsSettlement settlement = domainService.failSettlement(id, reason);
        return settlement.getId();
    }
    
    /**
     * 取消结算单
     */
    public Long cancelSettlement(Long id) {
        LogisticsSettlement settlement = domainService.cancelSettlement(id);
        return settlement.getId();
    }
    
    /**
     * 根据物流追踪号查询结算单
     */
    public LogisticsSettlement findByTrackingNo(String trackingNo) {
        return domainService.findByTrackingNo(trackingNo);
    }
    
    /**
     * 根据承运商查询结算单列表
     */
    public List<LogisticsSettlement> findByCarrierId(Long carrierId) {
        return domainService.findByCarrierId(carrierId);
    }
    
    /**
     * 根据状态查询结算单列表
     */
    public List<LogisticsSettlement> findByStatus(String status) {
        return domainService.findByStatus(
            LogisticsSettlement.LogisticsSettlementStatus.valueOf(status)
        );
    }
    
    /**
     * 结算单列表查询
     */
    public List<LogisticsSettlement> listAllSettlements() {
        return domainService.findByStatus(null);
    }
}