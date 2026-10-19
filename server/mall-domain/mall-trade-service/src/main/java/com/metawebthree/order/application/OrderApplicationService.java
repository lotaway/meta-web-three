package com.metawebthree.order.application;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.stream.Collectors;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.beans.factory.annotation.Value;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.common.exception.BusinessException;
import com.metawebthree.common.generated.rpc.GetProductBySkuRequest;
import com.metawebthree.common.generated.rpc.GetProductBySkuResponse;
import com.metawebthree.common.generated.rpc.GetProductDetailRequest;
import com.metawebthree.common.generated.rpc.GetProductDetailResponse;
import com.metawebthree.common.generated.rpc.ProductDetailProto;
import com.metawebthree.order.domain.model.OrderDO;
import com.metawebthree.order.domain.model.OrderItemDO;
import com.metawebthree.order.domain.ports.CommissionSettlementPort;
import com.metawebthree.order.domain.ports.PaymentRefundPort;
import com.metawebthree.order.infrastructure.client.InventoryClient;
import com.metawebthree.order.infrastructure.client.OrderPromotionClient;
import com.metawebthree.order.infrastructure.client.UserClient;
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
    private final OrderPromotionClient promotionClient;
    private final UserClient userClient;
    private final InventoryClient inventoryClient;
    private final PaymentRefundPort paymentRefundPort;
    @org.apache.dubbo.config.annotation.DubboReference(check = false, lazy = true)
    private com.metawebthree.common.generated.rpc.ProductService productService;

    @Value("${commission.return-window-days}")
    private int returnWindowDays;

    @Value("${inventory.default-warehouse-code:WH001}")
    private String defaultWarehouseCode;

    public OrderApplicationService(OrderMapper orderMapper, OrderItemMapper orderItemMapper,
            CommissionSettlementPort commissionSettlementPort,
            OrderEventPublisher eventPublisher,
            OrderPromotionClient promotionClient,
            UserClient userClient,
            InventoryClient inventoryClient,
            PaymentRefundPort paymentRefundPort) {
        this.orderMapper = orderMapper;
        this.orderItemMapper = orderItemMapper;
        this.commissionSettlementPort = commissionSettlementPort;
        this.eventPublisher = eventPublisher;
        this.promotionClient = promotionClient;
        this.userClient = userClient;
        this.inventoryClient = inventoryClient;
        this.paymentRefundPort = paymentRefundPort;
    }

    @Transactional
    public Long createOrder(Long userId, String remark, List<OrderItemCreate> items,
            Long memberReceiveAddressId, Long couponId, Integer useIntegration) {
        if (items == null || items.isEmpty()) {
            throw new BusinessException(ResponseStatus.PARAM_MISSING_ERROR, "Order must contain at least one item");
        }
        List<OrderItemDO> orderItems = new ArrayList<>();
        BigDecimal total = BigDecimal.ZERO;
        for (OrderItemCreate request : items) {
            ProductSnapshot snapshot = resolveProductSnapshot(request);
            BigDecimal itemTotal = snapshot.unitPrice.multiply(BigDecimal.valueOf(request.getQuantity()));
            total = total.add(itemTotal);
            orderItems.add(OrderItemDO.builder()
                    .id(IdWorker.getId())
                    .productId(snapshot.productId)
                    .productName(snapshot.productName)
                    .skuId(request.getSkuId())
                    .quantity(request.getQuantity())
                    .unitPrice(snapshot.unitPrice)
                    .totalPrice(itemTotal)
                    .imageUrl(snapshot.imageUrl)
                    .build());
        }
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
        List<Long> skuIds = items.stream().map(OrderItemCreate::getSkuId).collect(Collectors.toList());
        List<Integer> quantities = items.stream().map(OrderItemCreate::getQuantity).collect(Collectors.toList());
        boolean reserved = false;
        try {
            reserved = inventoryClient.reserveInventory(orderId, skuIds, quantities, "Order creation");
            if (!reserved) {
                throw new BusinessException(ResponseStatus.ORDER_CREATE_FAILED, "Insufficient inventory for order");
            }
            orderMapper.insert(order);
            for (OrderItemDO item : orderItems) {
                item.setOrderId(orderId);
                orderItemMapper.insert(item);
            }
            List<OrderEventPublisher.OrderItemCreate> eventItems = orderItems.stream()
                    .map(this::toEventItem)
                    .collect(Collectors.toList());
            eventPublisher.publishOrderCreated(orderId, orderNo, userId, total, "USD", eventItems);
            return orderId;
        } catch (RuntimeException e) {
            if (reserved) {
                releaseInventoryQuietly(orderId, "Order creation compensation");
            }
            throw e;
        }
    }

    private void releaseInventoryQuietly(Long orderId, String reason) {
        try {
            inventoryClient.releaseInventoryByOrderId(orderId, reason);
        } catch (Exception e) {
            log.error("Failed to release inventory for order {}: {}", orderId, e.getMessage(), e);
        }
    }

    private OrderEventPublisher.OrderItemCreate toEventItem(OrderItemDO item) {
        return new OrderEventPublisher.OrderItemCreate(
                item.getProductId(), item.getProductName(), item.getSkuId(),
                item.getQuantity(), item.getUnitPrice(), item.getImageUrl());
    }

    private ProductSnapshot resolveProductSnapshot(OrderItemCreate item) {
        if (item.getQuantity() == null || item.getQuantity() <= 0) {
            throw new BusinessException(ResponseStatus.PARAM_ERROR, "Item quantity must be positive");
        }
        ProductDetailProto product = queryProductSnapshot(item);
        if (product == null) {
            throw new BusinessException(ResponseStatus.PRODUCT_NOT_FOUND,
                    "Product not found for item productId=" + item.getProductId() + ", skuId=" + item.getSkuId());
        }
        if (product.getPrice() <= 0.0) {
            throw new BusinessException(ResponseStatus.PARAM_ERROR, "Product is not available for sale");
        }
        return new ProductSnapshot(product.getId(), product.getName(), product.getPic(),
                BigDecimal.valueOf(product.getPrice()));
    }

    private ProductDetailProto queryProductSnapshot(OrderItemCreate item) {
        if (item.getProductId() != null && item.getProductId() > 0) {
            GetProductDetailResponse detail = productService.getProductDetail(
                    GetProductDetailRequest.newBuilder().setProductId(item.getProductId()).build());
            if (detail != null && detail.hasProduct()) {
                return detail.getProduct();
            }
        }
        if (item.getSkuId() != null && item.getSkuId() > 0) {
            GetProductBySkuResponse bySku = productService.getProductBySku(
                    GetProductBySkuRequest.newBuilder().setSku(String.valueOf(item.getSkuId())).build());
            if (bySku != null && bySku.hasProduct()) {
                return bySku.getProduct();
            }
        }
        return null;
    }

    private static final class ProductSnapshot {
        private final Long productId;
        private final String productName;
        private final String imageUrl;
        private final BigDecimal unitPrice;

        ProductSnapshot(Long productId, String productName, String imageUrl, BigDecimal unitPrice) {
            this.productId = productId;
            this.productName = productName;
            this.imageUrl = imageUrl;
            this.unitPrice = unitPrice;
        }
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
        if ("CANCELED".equals(order.getOrderStatus())) {
            return true;
        }
        order.setOrderStatus("CANCELED");
        boolean updated = orderMapper.updateById(order) > 0;
        if (updated) {
            releaseInventoryQuietly(orderId, "Order canceled");
        }
        return updated;
    }

    @Transactional
    public boolean deleteOrder(Long orderId, Long userId) {
        OrderDO order = orderMapper.selectById(orderId);
        if (order == null || !order.getUserId().equals(userId)) {
            return false;
        }
        // 閫昏緫鍒犻櫎
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
        // Retrieve cart items from cart-service, address and points from user-service
        // Return a simple result for now
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
            
            grantIntegrationAndGrowth(order);
        }
        return updated;
    }

    private void grantIntegrationAndGrowth(OrderDO order) {
        if (order == null || order.getOrderAmount() == null) {
            return;
        }
        
        int integrationRate = 1;
        int growthRate = 1;
        
        int integrationToGrant = order.getOrderAmount().intValue() * integrationRate;
        int growthToGrant = order.getOrderAmount().intValue() * growthRate;
        
        if (integrationToGrant > 0) {
            Integer currentIntegration = userClient.addIntegration(
                    order.getUserId(), 
                    integrationToGrant, 
                    order.getId(), 
                    "Order completion reward points"
            );
            if (currentIntegration != null) {
                log.info("Order completion points reward - orderId: {}, userId: {}, integration: {}, current: {}", 
                        order.getId(), order.getUserId(), integrationToGrant, currentIntegration);
            }
        }
        
        if (growthToGrant > 0) {
            Integer currentGrowth = userClient.addGrowth(
                    order.getUserId(), 
                    growthToGrant, 
                    order.getId(), 
                    "Order completion reward growth"
            );
            if (currentGrowth != null) {
                log.info("Order completion growth reward - orderId: {}, userId: {}, growth: {}, current: {}", 
                        order.getId(), order.getUserId(), growthToGrant, currentGrowth);
            }
        }
    }

    @Transactional
    public boolean refundOrder(Long orderId, Long userId) {
        OrderDO order = orderMapper.selectById(orderId);
        if (order == null || !Objects.equals(order.getUserId(), userId)) {
            return false;
        }
        if ("REFUNDED".equals(order.getOrderStatus())) {
            // Idempotent: an already refunded order is treated as success.
            return true;
        }
        String previousStatus = order.getOrderStatus();
        if (!"PAID".equals(previousStatus) && !"COMPLETED".equals(previousStatus)) {
            // Unpaid / cancelled / failed orders cannot be refunded.
            return false;
        }

        // Claim the refund: only one concurrent caller may move the order into REFUNDING.
        int claimed = orderMapper.update(null, new LambdaUpdateWrapper<OrderDO>()
                .eq(OrderDO::getId, orderId)
                .in(OrderDO::getOrderStatus, "PAID", "COMPLETED")
                .set(OrderDO::getOrderStatus, "REFUNDING"));
        if (claimed == 0) {
            OrderDO latest = orderMapper.selectById(orderId);
            return latest != null && "REFUNDED".equals(latest.getOrderStatus());
        }

        boolean channelRefunded;
        try {
            channelRefunded = paymentRefundPort.refund(orderId, order.getOrderNo(), order.getOrderAmount(),
                    "User requested refund");
        } catch (RuntimeException e) {
            releaseRefundClaim(orderId, previousStatus);
            throw e;
        }
        if (!channelRefunded) {
            // The channel rejected the refund: restore the order so it can be retried.
            releaseRefundClaim(orderId, previousStatus);
            log.warn("Payment channel rejected refund for orderId: {}", orderId);
            return false;
        }

        int updated = orderMapper.update(null, new LambdaUpdateWrapper<OrderDO>()
                .eq(OrderDO::getId, orderId)
                .eq(OrderDO::getOrderStatus, "REFUNDING")
                .set(OrderDO::getOrderStatus, "REFUNDED"));
        if (updated > 0) {
            try {
                commissionSettlementPort.cancel(orderId);
            } catch (Exception e) {
                log.warn("commission cancel failed after refund: {}", e.getMessage());
            }
            return true;
        }
        log.error("Channel refund succeeded but order status update failed for orderId: {}", orderId);
        return false;
    }

    private void releaseRefundClaim(Long orderId, String previousStatus) {
        orderMapper.update(null, new LambdaUpdateWrapper<OrderDO>()
                .eq(OrderDO::getId, orderId)
                .eq(OrderDO::getOrderStatus, "REFUNDING")
                .set(OrderDO::getOrderStatus, previousStatus));
    }

    /**
     * Auto cancel timeout orders
     * Query orders that timeout without payment (status CREATED), update status to CANCELED
     * 
     * @param timeoutMinutes timeout in minutes
     * @return number of orders canceled
     */
    @Transactional
    public int cancelTimeOutOrder(int timeoutMinutes) {
        // Query orders that timeout without payment
        LocalDateTime timeoutTime = LocalDateTime.now().minusMinutes(timeoutMinutes);
        List<OrderDO> timeoutOrders = orderMapper.selectList(
                new LambdaQueryWrapper<OrderDO>()
                        .eq(OrderDO::getOrderStatus, "CREATED")
                        .lt(OrderDO::getCreatedAt, timeoutTime)
        );

        if (timeoutOrders.isEmpty()) {
            return 0;
        }

        // Batch update order status to CANCELED
        List<Long> orderIds = timeoutOrders.stream()
                .map(OrderDO::getId)
                .collect(Collectors.toList());

        for (OrderDO order : timeoutOrders) {
            order.setOrderStatus("CANCELED");
            orderMapper.updateById(order);
            // Execute order cancellation compensation
            processOrderCancellationCompensation(order);
        }

        return timeoutOrders.size();
    }

    /**
     * Handle order cancellation compensation
     * Include: release inventory lock, return coupon, return points
     * 
     * @param order canceled order
     */
    private void processOrderCancellationCompensation(OrderDO order) {
        log.info("Start processing order cancellation compensation - orderId: {}, orderNo: {}", order.getId(), order.getOrderNo());
        
        try {
            // 1. Release inventory lock - release inventory reserved by order
            // Release all reserved inventory for this order by order ID
            boolean inventoryReleased = inventoryClient.releaseInventoryByOrderId(
                    order.getId(), "Order cancellation inventory release");
            if (inventoryReleased) {
                log.info("Order inventory released successfully - orderId: {}", order.getId());
            } else {
                log.warn("Order inventory release failed - orderId: {}", order.getId());
            }
            
            // 2. Return coupon
            if (order.getCouponId() != null) {
                boolean couponReturned = promotionClient.returnCoupon(
                        order.getUserId(), order.getCouponId(), order.getId());
                if (couponReturned) {
                    log.info("Coupon returned successfully - userId: {}, couponId: {}, orderId: {}", 
                            order.getUserId(), order.getCouponId(), order.getId());
                } else {
                    log.warn("Coupon return failed - userId: {}, couponId: {}, orderId: {}", 
                            order.getUserId(), order.getCouponId(), order.getId());
                }
            } else {
                log.debug("Order did not use coupon, skip coupon return - orderId: {}", order.getId());
            }
            
            // 3. Return points
            if (order.getUseIntegration() != null && order.getUseIntegration() > 0) {
                Integer currentIntegration = userClient.returnIntegration(
                        order.getUserId(), order.getUseIntegration(), order.getId());
                if (currentIntegration != null) {
                    log.info("Points returned successfully - userId: {}, orderId: {}, returned points: {}, current points: {}", 
                            order.getUserId(), order.getId(), order.getUseIntegration(), currentIntegration);
                } else {
                    log.warn("Points return failed - userId: {}, orderId: {}, points: {}", 
                            order.getUserId(), order.getId(), order.getUseIntegration());
                }
            } else {
                log.debug("Order did not use points, skip points return - orderId: {}", order.getId());
            }
            
            log.info("Order cancellation compensation completed - orderId: {}", order.getId());
        } catch (Exception e) {
            log.error("Order cancellation compensation failed - orderId: {}, error: {}", order.getId(), e.getMessage(), e);
        }
    }

    @Data
    @io.swagger.v3.oas.annotations.media.Schema(description = "璁㈠崟鍟嗗搧鍒涘缓")
    public static class OrderItemCreate {
        @io.swagger.v3.oas.annotations.media.Schema(description = "鍟嗗搧ID")
        private Long productId;
        @io.swagger.v3.oas.annotations.media.Schema(description = "鍟嗗搧鍚嶇О")
        private String productName;
        @io.swagger.v3.oas.annotations.media.Schema(description = "SKU ID")
        private Long skuId;
        @io.swagger.v3.oas.annotations.media.Schema(description = "鏁伴噺")
        private Integer quantity;
        @io.swagger.v3.oas.annotations.media.Schema(description = "鍗曚环")
        private BigDecimal unitPrice;
        @io.swagger.v3.oas.annotations.media.Schema(description = "鍥剧墖URL")
        private String imageUrl;
    }

    @Data
    @io.swagger.v3.oas.annotations.media.Schema(description = "鍒涘缓璁㈠崟璇锋眰")
    public static class OrderCreateRequest {
        @io.swagger.v3.oas.annotations.media.Schema(description = "澶囨敞")
        private String remark;
        @io.swagger.v3.oas.annotations.media.Schema(description = "鍟嗗搧鍒楄〃")
        private List<OrderItemCreate> items;
        @io.swagger.v3.oas.annotations.media.Schema(description = "鏀惰揣鍦板潃ID")
        private Long memberReceiveAddressId;
        @io.swagger.v3.oas.annotations.media.Schema(description = "浼樻儬鍒窱D")
        private Long couponId;
        @io.swagger.v3.oas.annotations.media.Schema(description = "浣跨敤绉垎鏁伴噺")
        private Integer useIntegration;
    }

    @Data
    @io.swagger.v3.oas.annotations.media.Schema(description = "璁㈠崟璇︽儏(鍚晢鍝?")
    public static class OrderWithItems {
        @io.swagger.v3.oas.annotations.media.Schema(description = "璁㈠崟")
        private OrderDO order;
        @io.swagger.v3.oas.annotations.media.Schema(description = "鍟嗗搧鍒楄〃")
        private List<OrderItemDO> items;

        public BigDecimal getTotalPrice() {
            return items.stream().map(OrderItemDO::getTotalPrice).reduce(BigDecimal.ZERO, BigDecimal::add);
        }
    }
}

