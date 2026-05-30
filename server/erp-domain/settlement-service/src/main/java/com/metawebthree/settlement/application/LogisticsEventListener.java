package com.metawebthree.settlement.application;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.settlement.domain.service.LogisticsSettlementDomainService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.util.Map;

/**
 * 物流事件监听器
 * 监听物流服务发布的事件，自动触发运费结算
 */
@Slf4j
@Component
public class LogisticsEventListener {
    
    private final LogisticsSettlementApplicationService settlementService;
    private final ObjectMapper objectMapper;
    
    public LogisticsEventListener(LogisticsSettlementApplicationService settlementService,
                                   ObjectMapper objectMapper) {
        this.settlementService = settlementService;
        this.objectMapper = objectMapper;
    }
    
    /**
     * 监听物流订单送达事件，自动生成运费结算单
     */
    @KafkaListener(topics = "logistics.logistics.delivered", groupId = "settlement-service")
    public void onLogisticsDelivered(String message) {
        try {
            log.info("Received logistics.delivered event: {}", message);
            Map<String, Object> eventData = objectMapper.readValue(message, Map.class);
            
            String trackingNo = (String) eventData.get("trackingNo");
            if (trackingNo == null) {
                log.warn("trackingNo is null in logistics.delivered event");
                return;
            }
            
            // 从事件中获取物流订单信息
            // 实际项目中应该通过 RPC 调用物流服务获取完整信息
            // 这里简化处理，实际需要补充
            String orderNo = (String) eventData.get("orderNo");
            Long carrierId = eventData.get("carrierId") != null 
                ? ((Number) eventData.get("carrierId")).longValue() 
                : 1L;
            String carrierName = (String) eventData.get("carrierName");
            BigDecimal freight = eventData.get("freight") != null
                ? new BigDecimal(eventData.get("freight").toString())
                : BigDecimal.ZERO;
            
            // 检查是否已存在结算单，避免重复处理
            if (settlementService.findByTrackingNo(trackingNo) != null) {
                log.info("Settlement already exists for trackingNo: {}, skip", trackingNo);
                return;
            }
            
            // 自动创建运费结算单
            Long settlementId = settlementService.autoCreateSettlement(
                trackingNo, orderNo, carrierId, 
                carrierName != null ? carrierName : "Unknown Carrier",
                freight
            );
            
            log.info("Auto-created logistics settlement: trackingNo={}, settlementId={}", 
                    trackingNo, settlementId);
                    
        } catch (Exception e) {
            log.error("Failed to process logistics.delivered event: {}", message, e);
        }
    }
    
    /**
     * 监听物流订单创建事件（可选，用于记录或初始化）
     */
    @KafkaListener(topics = "logistics.logistics.created", groupId = "settlement-service")
    public void onLogisticsCreated(String message) {
        try {
            log.debug("Received logistics.created event: {}", message);
            // 物流订单创建时可以根据业务需求处理
            // 例如：预先创建结算单记录，或者只是记录日志
        } catch (Exception e) {
            log.error("Failed to process logistics.created event: {}", message, e);
        }
    }
}