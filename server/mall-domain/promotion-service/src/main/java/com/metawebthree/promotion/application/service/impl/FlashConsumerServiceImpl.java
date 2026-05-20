package com.metawebthree.promotion.application.service.impl;

import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.metawebthree.promotion.application.service.FlashConsumerService;
import com.metawebthree.promotion.infrastructure.persistence.mapper.*;
import com.metawebthree.promotion.infrastructure.persistence.model.*;
import com.metawebthree.promotion.interfaces.web.dto.FlashOrderRequest;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.util.*;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class FlashConsumerServiceImpl implements FlashConsumerService {

    private final FlashPromotionMapper flashPromotionMapper;
    private final FlashPromotionSessionMapper sessionMapper;
    private final FlashPromotionProductRelationMapper relationMapper;
    private final FlashPromotionLogMapper logMapper;
    private final FlashOrderMapper orderMapper;
    private final FlashOrderItemMapper orderItemMapper;
    private final FlashSkuStockMapper skuStockMapper;
    private final FlashProductMapper productMapper;

    @Override
    public List<Map<String, Object>> getCurrentPromotions() {
        LocalDate today = LocalDate.now();
        List<FlashPromotionDO> promotions = flashPromotionMapper.selectList(
                new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<FlashPromotionDO>()
                        .le(FlashPromotionDO::getStartDate, today)
                        .ge(FlashPromotionDO::getEndDate, today)
                        .eq(FlashPromotionDO::getStatus, 1));
        if (promotions.isEmpty()) {
            return Collections.emptyList();
        }
        LocalTime now = LocalTime.now();
        List<FlashPromotionSessionDO> allSessions = sessionMapper.selectList(
                new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<FlashPromotionSessionDO>()
                        .le(FlashPromotionSessionDO::getStartTime, now)
                        .ge(FlashPromotionSessionDO::getEndTime, now)
                        .eq(FlashPromotionSessionDO::getStatus, 1));
        if (allSessions.isEmpty()) {
            return Collections.emptyList();
        }
        Set<Long> sessionIds = allSessions.stream().map(FlashPromotionSessionDO::getId).collect(Collectors.toSet());
        List<FlashPromotionProductRelationDO> relations = relationMapper.selectList(
                new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<FlashPromotionProductRelationDO>()
                        .in(FlashPromotionProductRelationDO::getFlashPromotionSessionId, sessionIds));
        Map<Long, List<FlashPromotionProductRelationDO>> sessionProductMap = relations.stream()
                .collect(Collectors.groupingBy(FlashPromotionProductRelationDO::getFlashPromotionSessionId));
        return promotions.stream().map(promo -> {
            Map<String, Object> map = new HashMap<>();
            map.put("id", promo.getId());
            map.put("title", promo.getTitle());
            map.put("startDate", promo.getStartDate());
            map.put("endDate", promo.getEndDate());
            List<Map<String, Object>> sessionList = allSessions.stream().map(session -> {
                Map<String, Object> sm = new HashMap<>();
                sm.put("id", session.getId());
                sm.put("name", session.getName());
                sm.put("startTime", session.getStartTime());
                sm.put("endTime", session.getEndTime());
                List<FlashPromotionProductRelationDO> products = sessionProductMap.getOrDefault(session.getId(), Collections.emptyList());
                sm.put("products", products.stream().map(this::relationToMap).collect(Collectors.toList()));
                return sm;
            }).sorted(Comparator.comparing(s -> (LocalTime) s.get("startTime"))).collect(Collectors.toList());
            map.put("sessions", sessionList);
            return map;
        }).collect(Collectors.toList());
    }

    @Override
    public List<Map<String, Object>> getCurrentSessionProducts(Long sessionId) {
        LocalTime now = LocalTime.now();
        FlashPromotionSessionDO session = sessionMapper.selectById(sessionId);
        if (session == null || session.getStatus() != 1
                || now.isBefore(session.getStartTime()) || now.isAfter(session.getEndTime())) {
            return Collections.emptyList();
        }
        List<FlashPromotionProductRelationDO> relations = relationMapper.selectList(
                new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<FlashPromotionProductRelationDO>()
                        .eq(FlashPromotionProductRelationDO::getFlashPromotionSessionId, sessionId)
                        .gt(FlashPromotionProductRelationDO::getFlashPromotionCount, 0));
        return relations.stream().map(this::relationToMap).collect(Collectors.toList());
    }

    @Override
    public Map<String, Object> getProductFlashInfo(Long productId) {
        LocalDate today = LocalDate.now();
        LocalTime now = LocalTime.now();
        List<FlashPromotionDO> promotions = flashPromotionMapper.selectList(
                new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<FlashPromotionDO>()
                        .le(FlashPromotionDO::getStartDate, today)
                        .ge(FlashPromotionDO::getEndDate, today)
                        .eq(FlashPromotionDO::getStatus, 1));
        if (promotions.isEmpty()) {
            return Map.of("inFlash", false);
        }
        List<FlashPromotionSessionDO> sessions = sessionMapper.selectList(
                new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<FlashPromotionSessionDO>()
                        .le(FlashPromotionSessionDO::getStartTime, now)
                        .ge(FlashPromotionSessionDO::getEndTime, now)
                        .eq(FlashPromotionSessionDO::getStatus, 1));
        if (sessions.isEmpty()) {
            return Map.of("inFlash", false);
        }
        List<FlashPromotionProductRelationDO> relations = relationMapper.selectList(
                new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<FlashPromotionProductRelationDO>()
                        .eq(FlashPromotionProductRelationDO::getProductId, productId)
                        .in(FlashPromotionProductRelationDO::getFlashPromotionSessionId,
                                sessions.stream().map(FlashPromotionSessionDO::getId).collect(java.util.stream.Collectors.toList()))
                        .gt(FlashPromotionProductRelationDO::getFlashPromotionCount, 0));
        if (relations.isEmpty()) {
            return Map.of("inFlash", false);
        }
        FlashPromotionProductRelationDO r = relations.get(0);
        Map<String, Object> result = new HashMap<>();
        result.put("inFlash", true);
        result.put("sessionId", r.getFlashPromotionSessionId());
        result.put("flashPrice", r.getFlashPromotionPrice());
        result.put("flashCount", r.getFlashPromotionCount());
        result.put("flashLimit", r.getFlashPromotionLimit());
        return result;
    }

    @Override
    @Transactional
    public Long createFlashOrder(Long userId, FlashOrderRequest request) {
        LocalTime now = LocalTime.now();
        FlashPromotionSessionDO session = sessionMapper.selectById(request.getSessionId());
        if (session == null || session.getStatus() != 1
                || now.isBefore(session.getStartTime()) || now.isAfter(session.getEndTime())) {
            throw new IllegalArgumentException("闪购场次不在活动时间");
        }
        List<FlashPromotionProductRelationDO> relations = relationMapper.selectList(
                new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<FlashPromotionProductRelationDO>()
                        .eq(FlashPromotionProductRelationDO::getFlashPromotionSessionId, request.getSessionId())
                        .eq(FlashPromotionProductRelationDO::getProductId, request.getProductId()));
        if (relations.isEmpty()) {
            throw new IllegalArgumentException("该商品不在此闪购活动中");
        }
        FlashPromotionProductRelationDO relation = relations.get(0);
        if (relation.getFlashPromotionPrice().compareTo(request.getFlashPrice()) != 0) {
            throw new IllegalArgumentException("闪购价格不匹配");
        }
        if (request.getQuantity() > relation.getFlashPromotionLimit()) {
            throw new IllegalArgumentException("超过单次购买限制: " + relation.getFlashPromotionLimit());
        }
        int flashRows = skuStockMapper.deductFlashCount(relation.getId(), request.getQuantity());
        if (flashRows == 0) {
            throw new IllegalStateException("闪购库存不足");
        }
        int stockRows = skuStockMapper.deductStock(request.getSkuId(), request.getQuantity());
        if (stockRows == 0) {
            throw new IllegalStateException("商品库存不足");
        }
        BigDecimal unitPrice = relation.getFlashPromotionPrice();
        BigDecimal totalAmount = unitPrice.multiply(BigDecimal.valueOf(request.getQuantity()));
        LocalDateTime nowDt = LocalDateTime.now();
        Long orderId = IdWorker.getId();
        String orderSn = String.valueOf(IdWorker.getId());
        BigDecimal promotionAmount = BigDecimal.ZERO;
        orderMapper.insertOrder(orderId, userId, orderSn, totalAmount, totalAmount, promotionAmount,
                request.getOrderRemark(), "", "", "", "", "", "", null, nowDt);
        BigDecimal realAmount = totalAmount;
        orderItemMapper.insertOrderItem(IdWorker.getId(), orderId, request.getProductId(), request.getProductName(),
                request.getProductPic(), request.getSkuId(), request.getQuantity(),
                unitPrice, BigDecimal.ZERO, realAmount, nowDt);

        FlashPromotionLogDO logEntry = new FlashPromotionLogDO();
        logEntry.setMemberId(userId);
        logEntry.setProductId(request.getProductId());
        logEntry.setProductName(request.getProductName());
        logEntry.setSubscribeTime(nowDt);
        logMapper.insert(logEntry);

        log.info("Flash order created: orderId={}, userId={}, productId={}, quantity={}",
                orderId, userId, request.getProductId(), request.getQuantity());
        return orderId;
    }

    private Map<String, Object> relationToMap(FlashPromotionProductRelationDO r) {
        Map<String, Object> m = new HashMap<>();
        m.put("id", r.getId());
        m.put("flashPromotionId", r.getFlashPromotionId());
        m.put("flashPromotionSessionId", r.getFlashPromotionSessionId());
        m.put("productId", r.getProductId());
        m.put("flashPromotionPrice", r.getFlashPromotionPrice());
        m.put("flashPromotionCount", r.getFlashPromotionCount());
        m.put("flashPromotionLimit", r.getFlashPromotionLimit());
        m.put("sort", r.getSort());
        Map<String, Object> productInfo = productMapper.findProductInfo(r.getProductId());
        m.put("productName", productInfo != null ? productInfo.get("name") : null);
        m.put("productPic", productInfo != null ? productInfo.get("pic") : null);
        return m;
    }
}
