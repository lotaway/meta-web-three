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

@Service
public class OrderApplicationService {
    private final OrderMapper orderMapper;
    private final OrderItemMapper orderItemMapper;
    private final CommissionSettlementPort commissionSettlementPort;

    @Value("${commission.return-window-days}")
    private int returnWindowDays;

    public OrderApplicationService(OrderMapper orderMapper, OrderItemMapper orderItemMapper,
            CommissionSettlementPort commissionSettlementPort) {
        this.orderMapper = orderMapper;
        this.orderItemMapper = orderItemMapper;
        this.commissionSettlementPort = commissionSettlementPort;
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
        }

        // TODO: 恢复库存锁定、返还优惠券、返还积分等操作
        // 参考 temp/mall 的实现

        return timeoutOrders.size();
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
