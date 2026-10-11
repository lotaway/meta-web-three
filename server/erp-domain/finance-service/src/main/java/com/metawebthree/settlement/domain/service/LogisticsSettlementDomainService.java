package com.metawebthree.settlement.domain.service;

import com.metawebthree.settlement.domain.entity.LogisticsSettlement;
import com.metawebthree.settlement.domain.repository.LogisticsSettlementRepository;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.UUID;

/**
 * 物流费用结算领域服务
 * 处理物流运费自动结算的业务逻辑
 */
@Service
public class LogisticsSettlementDomainService {
    
    private final LogisticsSettlementRepository repository;
    
    // 默认手续费率 1%
    private static final BigDecimal DEFAULT_HANDLING_FEE_RATE = new BigDecimal("0.01");
    
    public LogisticsSettlementDomainService(LogisticsSettlementRepository repository) {
        this.repository = repository;
    }
    
    /**
     * 创建物流运费结算单（物流订单送达时调用）
     * @param trackingNo 物流追踪号
     * @param orderNo 订单号
     * @param carrierId 承运商ID
     * @param carrierName 承运商名称
     * @param freight 运费
     * @return 结算单
     */
    public LogisticsSettlement createSettlement(String trackingNo, String orderNo,
                                                  Long carrierId, String carrierName,
                                                  BigDecimal freight) {
        return createSettlement(trackingNo, orderNo, carrierId, carrierName, freight, 
                                DEFAULT_HANDLING_FEE_RATE, BigDecimal.ZERO);
    }
    
    /**
     * 创建物流运费结算单（可指定手续费率和折扣）
     */
    public LogisticsSettlement createSettlement(String trackingNo, String orderNo,
                                                  Long carrierId, String carrierName,
                                                  BigDecimal freight, 
                                                  BigDecimal handlingFeeRate,
                                                  BigDecimal discount) {
        // 检查是否已存在该物流单的结算记录
        if (repository.findByTrackingNo(trackingNo).isPresent()) {
            throw new IllegalStateException("Settlement already exists for trackingNo: " + trackingNo);
        }
        
        String settlementNo = generateSettlementNo();
        LogisticsSettlement settlement = new LogisticsSettlement();
        settlement.createLogisticsSettlement(
            settlementNo, trackingNo, orderNo, 
            carrierId, carrierName, freight,
            handlingFeeRate, discount
        );
        
        return repository.save(settlement);
    }
    
    /**
     * 确认结算单
     */
    public LogisticsSettlement confirmSettlement(Long id) {
        LogisticsSettlement settlement = repository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("Settlement not found: " + id));
        settlement.confirm();
        return repository.save(settlement);
    }
    
    /**
     * 处理结算（执行付款）
     */
    public LogisticsSettlement processSettlement(Long id) {
        LogisticsSettlement settlement = repository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("Settlement not found: " + id));
        settlement.process();
        return repository.save(settlement);
    }
    
    /**
     * 完成结算（付款成功）
     */
    public LogisticsSettlement completeSettlement(Long id) {
        LogisticsSettlement settlement = repository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("Settlement not found: " + id));
        settlement.complete();
        return repository.save(settlement);
    }
    
    /**
     * 结算失败
     */
    public LogisticsSettlement failSettlement(Long id, String reason) {
        LogisticsSettlement settlement = repository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("Settlement not found: " + id));
        settlement.fail(reason);
        return repository.save(settlement);
    }
    
    /**
     * 取消结算单
     */
    public LogisticsSettlement cancelSettlement(Long id) {
        LogisticsSettlement settlement = repository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("Settlement not found: " + id));
        settlement.cancel();
        return repository.save(settlement);
    }
    
    /**
     * 根据物流追踪号查询结算单
     */
    public LogisticsSettlement findByTrackingNo(String trackingNo) {
        return repository.findByTrackingNo(trackingNo).orElse(null);
    }
    
    /**
     * 根据承运商查询结算单列表
     */
    public java.util.List<LogisticsSettlement> findByCarrierId(Long carrierId) {
        return repository.findByCarrierId(carrierId);
    }
    
    /**
     * 根据状态查询结算单列表
     */
    public java.util.List<LogisticsSettlement> findByStatus(LogisticsSettlement.LogisticsSettlementStatus status) {
        return repository.findByStatus(status);
    }
    
    /**
     * 生成结算单号
     * 格式: LGS + 年月日 + UUID前8位
     */
    private String generateSettlementNo() {
        String date = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd"));
        String uuid = UUID.randomUUID().toString().substring(0, 8).toUpperCase();
        return "LGS" + date + uuid;
    }
}