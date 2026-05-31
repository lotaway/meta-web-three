package com.metawebthree.order.application;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.beans.factory.annotation.Value;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.metawebthree.order.domain.model.OrderDO;
import com.metawebthree.order.domain.model.OrderItemDO;
import com.metawebthree.order.domain.ports.CommissionSettlementPort;
import com.metawebthree.order.infrastructure.persistence.mapper.OrderItemMapper;
import com.metawebthree.order.infrastructure.persistence.mapper.OrderMapper;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@Service
public class OrderApplicationService {
    private final OrderMapper orderMapper;
    private final OrderItemMapper orderItemMapper;
    private final CommissionSettlementPort commissionSettlementPort;
    private final OrderEventPublisher eventPublisher;

    @Value("${commission.return-window-days}")
    private int returnWindowDays;

    public OrderApplicationService(OrderMapper orderMapper, OrderItemMapper orderItemMapper,
            CommissionSettlementPort commissionSettlementPort,
            OrderEventPublisher eventPublisher) {
        this.orderMapper = orderMapper;
        this.orderItemMapper = orderItemMapper;
        this.commissionSettlementPort = commissionSettlementPort;
        this.eventPublisher = eventPublisher;
    }

    @Transactional
    public Long createOrder(Long userId, String remark, List<OrderItemCreate> items,
            Long memberReceiveAddressId, Long couponId, Integer useIntegration) {
        BigDecimal total = items.stream()
                .map(i -> i.getUnitPrice().multiply(BigDecimal.valueOf(i.getQuantity())))
                .reduce(BigDecimal.ZERO, BigDecimal::add);
        Long orderId = IdWorker.getId();
        String orderNo = String.valueOf(IdWorker.getId());
        OrderDO order = OrderDO.builder()
                .id(orderId)
                .userId(userId)
                .orderNo(orderNo)
                .orderStatus("CREATED")
                .orderType("NORMAL")
                .orderAmount(total)
                .orderRemark(remark)
                .memberReceiveAddressId(memberReceiveAddressId)
                .couponId(couponId)
                .useIntegration(useIntegration)
                .build();
        orderMapper.insert(order);
        for (OrderItemCreate i : items) {
            BigDecimal totalPrice = i.getUnitPrice().multiply(BigDecimal.valueOf(i.getQuantity()));
            OrderItemDO item = OrderItemDO.builder()
                    .id(IdWorker.getId())
                    .orderId(orderId)
                    .productId(i.getProductId())
                    .productName(i.getProductName())
                    .skuId(i.getSkuId())
                    .quantity(i.getQuantity())
                    .unitPrice(i.getUnitPrice())
                    .totalPrice(totalPrice)
                    .imageUrl(i.getImageUrl())
                    .build();
            orderItemMapper.insert(item);
        }
        // Publish order created event
        List<OrderEventPublisher.OrderItemCreate> eventItems = items.stream()
                .map(i -> new OrderEventPublisher.OrderItemCreate(
                        i.getProductId(), i.getProductName(), i.getSkuId(),
                        i.getQuantity(), i.getUnitPrice(), i.getImageUrl()))
                .collect(Collectors.toList());
        eventPublisher.publishOrderCreated(orderId, orderNo, userId, total, "USD", eventItems);
        return orderId;
    }

    public Optional<OrderWithItems> getOrderDetail(Long orderId, Long userId) {
        OrderDO order = orderMapper.selectById(orderId);
        if (order == null || (userId != null && !order.getUserId().equals(userId))) {
            return Optional.empty();
        }
        List<OrderItemDO> items = orderItemMapper.selectList(new LambdaQueryWrapper<OrderItemDO>()
                .eq(OrderItemDO::getOrderId, orderId));
        OrderWithItems detail = new OrderWithItems();
        detail.setOrder(order);
        detail.setItems(items);
        return Optional.of(detail);
    }

    public List<OrderDO> listOrdersByUser(Long userId, int pageNum, int pageSize) {
        return orderMapper.selectList(new LambdaQueryWrapper<OrderDO>()
                .eq(OrderDO::getUserId, userId)
                .last("limit " + pageSize + " offset " + Math.max(0, (pageNum - 1) * pageSize)));
    }

    @Transactional
    public boolean cancelOrder(Long orderId, Long userId) {
        OrderDO order = orderMapper.selectById(orderId);
        if (order == null || !order.getUserId().equals(userId)) {
            return false;
        }
        order.setOrderStatus("CANCELED");
        return orderMapper.updateById(order) > 0;
    }

    @Transactional
    public boolean deleteOrder(Long orderId, Long userId) {
        OrderDO order = orderMapper.selectById(orderId);
        if (order == null || !order.getUserId().equals(userId)) {
            return false;
        }
        // 逻辑删除
        order.setDeleteStatus(1);
        return orderMapper.updateById(order) > 0;
    }

    @Transactional
    public boolean paySuccess(Long orderId, Integer payType) {
        OrderDO order = orderMapper.selectById(orderId);
        if (order == null) {
            return false;
        }
        order.setOrderStatus("PAID");
        order.setPaymentTime(LocalDateTime.now());
        order.setPaymentType(payType);
        return orderMapper.updateById(order) > 0;
    }

    public ConfirmOrderResult generateConfirmOrder(Long userId, List<Long> cartIds) {
        // 这里应该是从 cart-service 获取购物车项，从 user-service 获取地址和积分等
        // 目前先返回一个简单的结果
        ConfirmOrderResult result = new ConfirmOrderResult();
        result.setCartPromotionItemList(Collections.emptyList());
        result.setMemberReceiveAddressList(Collections.emptyList());
        result.setMemberCouponList(Collections.emptyList());
        result.setCalcAmount(new CalcAmount());
        return result;
    }

    @Data
    public static class ConfirmOrderResult {
        private List<?> cartPromotionItemList;
        private List<?> memberReceiveAddressList;
        private List<?> memberCouponList;
        private CalcAmount calcAmount;
    }

    @Data
    public static class CalcAmount {
        private BigDecimal totalAmount = BigDecimal.ZERO;
        private BigDecimal freightAmount = BigDecimal.ZERO;
        private BigDecimal promotionAmount = BigDecimal.ZERO;
        private BigDecimal payAmount = BigDecimal.ZERO;
    }

    @Transactional
    public boolean confirmReceive(Long orderId, Long userId) {
        OrderDO order = orderMapper.selectById(orderId);
        if (order == null || !order.getUserId().equals(userId)) {
            return false;
        }
        order.setOrderStatus("COMPLETED");
        boolean updated = orderMapper.updateById(order) > 0;
        if (updated) {
            LocalDateTime confirmAt = LocalDateTime.now();
            LocalDateTime returnDeadlineAt = confirmAt.plusDays(returnWindowDays);
            LocalDateTime availableAt = returnDeadlineAt.plusDays(1);
            commissionSettlementPort.calculate(orderId, userId, order.getOrderAmount(), availableAt);
        }
        return updated;
    }

    @Transactional
    public boolean refundOrder(Long orderId, Long userId) {
        OrderDO order = orderMapper.selectById(orderId);
        if (order == null || !order.getUserId().equals(userId)) {
            return false;
        }
        order.setOrderStatus("REFUNDED");
        boolean updated = orderMapper.updateById(order) > 0;
        if (updated) {
            commissionSettlementPort.cancel(orderId);
        }
        return updated;
    }

    /**
     * 自动取消超时订单
     * 查询超时未支付的订单（状态为CREATED），并更新状态为CANCELED
     * 
     * @param timeoutMinutes 超时时间（分钟）
     * @return 取消的订单数量
     */
    @Transactional
    public int cancelTimeOutOrder(int timeoutMinutes) {
        // 查询超时未支付订单
        LocalDateTime timeoutTime = LocalDateTime.now().minusMinutes(timeoutMinutes);
        List<OrderDO> timeoutOrders = orderMapper.selectList(
                new LambdaQueryWrapper<OrderDO>()
                        .eq(OrderDO::getOrderStatus, "CREATED")
                        .lt(OrderDO::getCreatedAt, timeoutTime)
        );

        if (timeoutOrders.isEmpty()) {
            return 0;
        }

        // 批量更新订单状态为CANCELED
        List<Long> orderIds = timeoutOrders.stream()
                .map(OrderDO::getId)
                .collect(Collectors.toList());

        for (OrderDO order : timeoutOrders) {
            order.setOrderStatus("CANCELED");
            orderMapper.updateById(order);
            // 执行订单取消补偿
            processOrderCancellationCompensation(order);
        }

        return timeoutOrders.size();
    }

    /**
     * 处理订单取消补偿
     * 包括：恢复库存锁定、返还优惠券、返还积分等操作
     * 
     * @param order 取消的订单
     */
    private void processOrderCancellationCompensation(OrderDO order) {
        log.info("开始处理订单取消补偿 - 订单ID: {}, 订单号: {}", order.getId(), order.getOrderNo());
        
        try {
            // 1. 恢复库存锁定
            // TODO: 调用 inventory-service 释放订单占用的库存
            // inventoryService.releaseStock(order.getId());
            log.debug("恢复库存锁定 - 待集成 inventory-service");
            
            // 2. 返还优惠券
            // TODO: 调用 promotion-service 返还用户使用的优惠券
            // if (order.getCouponId() != null) {
            //     promotionService.returnCoupon(order.getUserId(), order.getCouponId());
            // }
            log.debug("返还优惠券 - 待集成 promotion-service, couponId: {}", order.getCouponId());
            
            // 3. 返还积分
            // TODO: 调用 user-service 返还用户使用的积分
            // if (order.getUseIntegration() != null && order.getUseIntegration() > 0) {
            //     userService.returnIntegration(order.getUserId(), order.getUseIntegration());
            // }
            log.debug("返还积分 - 待集成 user-service, integration: {}", order.getUseIntegration());
            
            log.info("订单取消补偿处理完成 - 订单ID: {}", order.getId());
        } catch (Exception e) {
            log.error("订单取消补偿处理失败 - 订单ID: {}, 错误: {}", order.getId(), e.getMessage(), e);
        }
    }

    @Data
    @io.swagger.v3.oas.annotations.media.Schema(description = "订单商品创建")
    public static class OrderItemCreate {
        @io.swagger.v3.oas.annotations.media.Schema(description = "商品ID")
        private Long productId;
        @io.swagger.v3.oas.annotations.media.Schema(description = "商品名称")
        private String productName;
        @io.swagger.v3.oas.annotations.media.Schema(description = "SKU ID")
        private Long skuId;
        @io.swagger.v3.oas.annotations.media.Schema(description = "数量")
        private Integer quantity;
        @io.swagger.v3.oas.annotations.media.Schema(description = "单价")
        private BigDecimal unitPrice;
        @io.swagger.v3.oas.annotations.media.Schema(description = "图片URL")
        private String imageUrl;
    }

    @Data
    @io.swagger.v3.oas.annotations.media.Schema(description = "创建订单请求")
    public static class OrderCreateRequest {
        @io.swagger.v3.oas.annotations.media.Schema(description = "备注")
        private String remark;
        @io.swagger.v3.oas.annotations.media.Schema(description = "商品列表")
        private List<OrderItemCreate> items;
        @io.swagger.v3.oas.annotations.media.Schema(description = "收货地址ID")
        private Long memberReceiveAddressId;
        @io.swagger.v3.oas.annotations.media.Schema(description = "优惠券ID")
        private Long couponId;
        @io.swagger.v3.oas.annotations.media.Schema(description = "使用积分数量")
        private Integer useIntegration;
    }

    @Data
    @io.swagger.v3.oas.annotations.media.Schema(description = "订单详情(含商品)")
    public static class OrderWithItems {
        @io.swagger.v3.oas.annotations.media.Schema(description = "订单")
        private OrderDO order;
        @io.swagger.v3.oas.annotations.media.Schema(description = "商品列表")
        private List<OrderItemDO> items;

        public BigDecimal getTotalPrice() {
            return items.stream().map(OrderItemDO::getTotalPrice).reduce(BigDecimal.ZERO, BigDecimal::add);
        }
    }
}
