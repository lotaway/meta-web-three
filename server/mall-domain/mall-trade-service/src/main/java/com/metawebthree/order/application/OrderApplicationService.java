package com.metawebthree.order.application;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.beans.factory.annotation.Value;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.metawebthree.cart.application.CartService;
import com.metawebthree.cart.dto.CartItemDTO;
import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.common.exception.BusinessException;
import com.metawebthree.common.generated.rpc.GetProductBySkuRequest;
import com.metawebthree.common.generated.rpc.GetProductBySkuResponse;
import com.metawebthree.common.generated.rpc.GetProductDetailRequest;
import com.metawebthree.common.generated.rpc.GetProductDetailResponse;
import com.metawebthree.common.generated.rpc.MemberAddressProto;
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
import com.metawebthree.promotion.application.CouponCommandService;
import com.metawebthree.promotion.application.CouponQueryService;
import com.metawebthree.promotion.domain.enums.CouponStatus;
import com.metawebthree.promotion.domain.model.Coupon;
import com.metawebthree.promotion.domain.model.CouponType;

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
    private final CartService cartService;
    private final CouponQueryService couponQueryService;
    private final CouponCommandService couponCommandService;
    @org.apache.dubbo.config.annotation.DubboReference(check = false, lazy = true)
    private com.metawebthree.common.generated.rpc.ProductService productService;

    @Value("${commission.return-window-days}")
    private int returnWindowDays;

    @Value("${inventory.default-warehouse-code:WH001}")
    private String defaultWarehouseCode;

    /** How many membership points are worth one unit of currency. */
    @Value("${order.points-per-currency-unit:100}")
    private int pointsPerCurrencyUnit;

    public OrderApplicationService(OrderMapper orderMapper, OrderItemMapper orderItemMapper,
            CommissionSettlementPort commissionSettlementPort,
            OrderEventPublisher eventPublisher,
            OrderPromotionClient promotionClient,
            UserClient userClient,
            InventoryClient inventoryClient,
            PaymentRefundPort paymentRefundPort,
            CartService cartService,
            CouponQueryService couponQueryService,
            CouponCommandService couponCommandService) {
        this.orderMapper = orderMapper;
        this.orderItemMapper = orderItemMapper;
        this.commissionSettlementPort = commissionSettlementPort;
        this.eventPublisher = eventPublisher;
        this.promotionClient = promotionClient;
        this.userClient = userClient;
        this.inventoryClient = inventoryClient;
        this.paymentRefundPort = paymentRefundPort;
        this.cartService = cartService;
        this.couponQueryService = couponQueryService;
        this.couponCommandService = couponCommandService;
    }

    @Transactional
    public Long createOrder(Long userId, String remark, List<OrderItemCreate> items,
            Long memberReceiveAddressId, Long couponId, Integer useIntegration) {
        if (items == null || items.isEmpty()) {
            throw new BusinessException(ResponseStatus.PARAM_MISSING_ERROR, "Order must contain at least one item");
        }
        if (userId == null) {
            throw new BusinessException(ResponseStatus.PARAM_MISSING_ERROR, "Missing userId");
        }

        // The delivery address must belong to the buyer. Fail closed when the user
        // service is unreachable so an order is never shipped to a foreign address.
        if (memberReceiveAddressId != null) {
            boolean owned = userClient.listMemberAddresses(userId).stream()
                    .anyMatch(address -> address.getId() == memberReceiveAddressId);
            if (!owned) {
                throw new BusinessException(ResponseStatus.NOT_FOUND,
                        "Delivery address does not belong to the current user");
            }
        }

        List<PricedItem> pricedItems = new ArrayList<>();
        BigDecimal subtotal = BigDecimal.ZERO;
        for (OrderItemCreate request : items) {
            PricedItem priced = priceLine(request);
            pricedItems.add(priced);
            subtotal = subtotal.add(priced.lineTotal);
        }

        Discounts discounts = applyDiscounts(userId, subtotal, couponId, useIntegration);
        BigDecimal payable = discounts.payAmount;

        List<OrderItemDO> orderItems = pricedItems.stream()
                .map(priced -> OrderItemDO.builder()
                        .id(IdWorker.getId())
                        .productId(priced.productId)
                        .productName(priced.productName)
                        .skuId(priced.skuId)
                        .quantity(priced.quantity)
                        .unitPrice(priced.unitPrice)
                        .totalPrice(priced.lineTotal)
                        .imageUrl(priced.imageUrl)
                        .build())
                .collect(Collectors.toList());

        Long orderId = IdWorker.getId();
        String orderNo = String.valueOf(IdWorker.getId());
        OrderDO order = OrderDO.builder()
                .id(orderId)
                .userId(userId)
                .orderNo(orderNo)
                .orderStatus("CREATED")
                .orderType("NORMAL")
                .orderAmount(payable)
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
            if (discounts.coupon != null) {
                // Consume the coupon in the same transaction so it cannot be reused
                // for another order; cancellation returns it (compensation path).
                couponCommandService.consume(discounts.coupon.getCode(), userId, orderNo, null, null);
            }
            List<OrderEventPublisher.OrderItemCreate> eventItems = orderItems.stream()
                    .map(this::toEventItem)
                    .collect(Collectors.toList());
            eventPublisher.publishOrderCreated(orderId, orderNo, userId, payable, "USD", eventItems);
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
        return generateConfirmOrder(userId, cartIds, null, null);
    }

    /**
     * Build the checkout preview for the selected cart items using the same
     * server-side pricing and discount rules as {@link #createOrder}, so the
     * confirmed amount always equals the amount stored on the order.
     */
    public ConfirmOrderResult generateConfirmOrder(Long userId, List<Long> cartIds, Long couponId,
            Integer useIntegration) {
        if (userId == null) {
            throw new BusinessException(ResponseStatus.PARAM_MISSING_ERROR, "Missing userId");
        }
        List<CartItemDTO> cartItems = cartService.list(userId);
        if (cartIds != null && !cartIds.isEmpty()) {
            Set<Long> wanted = new HashSet<>(cartIds);
            cartItems = cartItems.stream()
                    .filter(item -> wanted.contains(item.getId()))
                    .collect(Collectors.toList());
        }
        if (cartItems == null || cartItems.isEmpty()) {
            throw new BusinessException(ResponseStatus.PARAM_ERROR, "No cart items selected for checkout");
        }

        List<ConfirmOrderItem> items = new ArrayList<>();
        BigDecimal subtotal = BigDecimal.ZERO;
        for (CartItemDTO cartItem : cartItems) {
            Integer quantity = cartItem.getQuantity();
            if (quantity == null || quantity <= 0) {
                throw new BusinessException(ResponseStatus.PARAM_ERROR, "Invalid cart item quantity");
            }
            Long stockKey = cartItem.getProductSkuId() != null ? cartItem.getProductSkuId()
                    : cartItem.getProductId();
            assertStockAvailable(stockKey, quantity);

            OrderItemCreate probe = new OrderItemCreate();
            probe.setProductId(cartItem.getProductId());
            probe.setSkuId(cartItem.getProductSkuId());
            probe.setQuantity(quantity);
            PricedItem priced = priceLine(probe);
            ConfirmOrderItem item = new ConfirmOrderItem();
            item.setCartItemId(cartItem.getId());
            item.setProductId(priced.productId);
            item.setSkuId(priced.skuId);
            item.setProductName(priced.productName);
            item.setImageUrl(priced.imageUrl);
            item.setQuantity(priced.quantity);
            item.setUnitPrice(priced.unitPrice);
            item.setTotalPrice(priced.lineTotal);
            item.setPriceChanged(cartItem.getPrice() != null
                    && cartItem.getPrice().compareTo(priced.unitPrice) != 0);
            items.add(item);
            subtotal = subtotal.add(priced.lineTotal);
        }

        Discounts discounts = applyDiscounts(userId, subtotal, couponId, useIntegration);
        List<ConfirmAddress> addresses = userClient.listMemberAddresses(userId).stream()
                .map(this::toConfirmAddress)
                .collect(Collectors.toList());
        List<Coupon> coupons = couponQueryService.listByOwner(userId, CouponStatus.UNUSED.getCode());
        Integer integration = userClient.getUserIntegration(userId);

        ConfirmOrderResult result = new ConfirmOrderResult();
        result.setCartPromotionItemList(items);
        result.setMemberReceiveAddressList(addresses);
        result.setMemberCouponList(coupons);
        result.setMemberIntegration(integration);
        CalcAmount calc = new CalcAmount();
        calc.setTotalAmount(subtotal);
        calc.setPromotionAmount(discounts.couponDiscount);
        calc.setIntegrationAmount(discounts.pointsDiscount);
        calc.setFreightAmount(BigDecimal.ZERO);
        calc.setPayAmount(discounts.payAmount);
        result.setCalcAmount(calc);
        return result;
    }

    /**
     * Resolve the authoritative price of a single line from the product service.
     * Cart-supplied prices are never trusted.
     */
    private PricedItem priceLine(OrderItemCreate request) {
        ProductSnapshot snapshot = resolveProductSnapshot(request);
        BigDecimal lineTotal = snapshot.unitPrice.multiply(BigDecimal.valueOf(request.getQuantity()));
        return new PricedItem(snapshot.productId, snapshot.productName, snapshot.imageUrl,
                request.getSkuId(), request.getQuantity(), snapshot.unitPrice, lineTotal);
    }

    private Discounts applyDiscounts(Long userId, BigDecimal subtotal, Long couponId, Integer useIntegration) {
        Coupon coupon = null;
        BigDecimal couponDiscount = BigDecimal.ZERO;
        if (couponId != null) {
            coupon = couponQueryService.listByOwner(userId, CouponStatus.UNUSED.getCode()).stream()
                    .filter(c -> Objects.equals(c.getId(), couponId))
                    .findFirst()
                    .orElseThrow(() -> new BusinessException(ResponseStatus.NOT_FOUND,
                            "Coupon not found, not owned by the user or already used"));
            couponDiscount = validateAndComputeCouponDiscount(coupon, subtotal);
        }
        BigDecimal afterCoupon = subtotal.subtract(couponDiscount).max(BigDecimal.ZERO);
        BigDecimal pointsDiscount = resolvePointsDiscount(userId, useIntegration, afterCoupon);
        BigDecimal payAmount = afterCoupon.subtract(pointsDiscount).max(BigDecimal.ZERO);
        return new Discounts(coupon, couponDiscount, pointsDiscount, payAmount);
    }

    private BigDecimal validateAndComputeCouponDiscount(Coupon coupon, BigDecimal subtotal) {
        CouponType type = couponQueryService.getCouponType(coupon.getCouponTypeId());
        if (type == null) {
            throw new BusinessException(ResponseStatus.NOT_FOUND, "Coupon type not found");
        }
        if (!Boolean.TRUE.equals(type.getIsEnabled())) {
            throw new BusinessException(ResponseStatus.PARAM_ERROR, "Coupon is disabled");
        }
        LocalDateTime now = LocalDateTime.now();
        if (type.getStartTime() != null && type.getStartTime().isAfter(now)) {
            throw new BusinessException(ResponseStatus.PARAM_ERROR, "Coupon is not yet valid");
        }
        if (type.getEndTime() != null && type.getEndTime().isBefore(now)) {
            throw new BusinessException(ResponseStatus.PARAM_ERROR, "Coupon has expired");
        }
        if (type.getMinimumOrderAmount() != null && subtotal.compareTo(type.getMinimumOrderAmount()) < 0) {
            throw new BusinessException(ResponseStatus.PARAM_ERROR,
                    "Order amount does not meet the coupon minimum of " + type.getMinimumOrderAmount());
        }
        BigDecimal discount = type.getDiscountAmount() != null ? type.getDiscountAmount() : BigDecimal.ZERO;
        return discount.min(subtotal);
    }

    private BigDecimal resolvePointsDiscount(Long userId, Integer useIntegration, BigDecimal ceiling) {
        if (useIntegration == null || useIntegration <= 0) {
            return BigDecimal.ZERO;
        }
        if (pointsPerCurrencyUnit <= 0 || ceiling.compareTo(BigDecimal.ZERO) <= 0) {
            return BigDecimal.ZERO;
        }
        Integer balance = userClient.getUserIntegration(userId);
        if (balance == null) {
            throw new BusinessException(ResponseStatus.SYSTEM_ERROR, "Failed to verify membership points");
        }
        if (balance < useIntegration) {
            throw new BusinessException(ResponseStatus.PAYMENT_INSUFFICIENT_BALANCE,
                    "Insufficient membership points: requested " + useIntegration + ", available " + balance);
        }
        BigDecimal discount = BigDecimal.valueOf(useIntegration)
                .divide(BigDecimal.valueOf(pointsPerCurrencyUnit), 2, RoundingMode.DOWN);
        return discount.min(ceiling);
    }

    private void assertStockAvailable(Long productId, Integer quantity) {
        if (productId == null || productId <= 0) {
            return;
        }
        Integer available = inventoryClient.getAvailableQuantity(productId, null);
        if (available != null && available < quantity) {
            throw new BusinessException(ResponseStatus.ORDER_CREATE_FAILED,
                    "Insufficient inventory for product " + productId
                            + ": required " + quantity + ", available " + available);
        }
    }

    private ConfirmAddress toConfirmAddress(MemberAddressProto address) {
        ConfirmAddress confirm = new ConfirmAddress();
        confirm.setId(address.getId());
        confirm.setName(address.getName());
        confirm.setPhoneNumber(address.getPhoneNumber());
        confirm.setDefaultStatus(address.getDefaultStatus());
        confirm.setPostCode(address.getPostCode());
        confirm.setProvince(address.getProvince());
        confirm.setCity(address.getCity());
        confirm.setRegion(address.getRegion());
        confirm.setDetailAddress(address.getDetailAddress());
        return confirm;
    }

    private static final class PricedItem {
        private final Long productId;
        private final String productName;
        private final String imageUrl;
        private final Long skuId;
        private final Integer quantity;
        private final BigDecimal unitPrice;
        private final BigDecimal lineTotal;

        PricedItem(Long productId, String productName, String imageUrl, Long skuId,
                Integer quantity, BigDecimal unitPrice, BigDecimal lineTotal) {
            this.productId = productId;
            this.productName = productName;
            this.imageUrl = imageUrl;
            this.skuId = skuId;
            this.quantity = quantity;
            this.unitPrice = unitPrice;
            this.lineTotal = lineTotal;
        }
    }

    private static final class Discounts {
        private final Coupon coupon;
        private final BigDecimal couponDiscount;
        private final BigDecimal pointsDiscount;
        private final BigDecimal payAmount;

        Discounts(Coupon coupon, BigDecimal couponDiscount, BigDecimal pointsDiscount, BigDecimal payAmount) {
            this.coupon = coupon;
            this.couponDiscount = couponDiscount;
            this.pointsDiscount = pointsDiscount;
            this.payAmount = payAmount;
        }
    }

    @Data
    public static class ConfirmOrderResult {
        private List<ConfirmOrderItem> cartPromotionItemList;
        private List<ConfirmAddress> memberReceiveAddressList;
        private List<Coupon> memberCouponList;
        private Integer memberIntegration;
        private CalcAmount calcAmount;
    }

    @Data
    public static class ConfirmOrderItem {
        private Long cartItemId;
        private Long productId;
        private Long skuId;
        private String productName;
        private String imageUrl;
        private Integer quantity;
        private BigDecimal unitPrice;
        private BigDecimal totalPrice;
        private boolean priceChanged;
    }

    @Data
    public static class ConfirmAddress {
        private Long id;
        private String name;
        private String phoneNumber;
        private boolean defaultStatus;
        private String postCode;
        private String province;
        private String city;
        private String region;
        private String detailAddress;
    }

    @Data
    public static class CalcAmount {
        private BigDecimal totalAmount = BigDecimal.ZERO;
        private BigDecimal freightAmount = BigDecimal.ZERO;
        private BigDecimal promotionAmount = BigDecimal.ZERO;
        private BigDecimal integrationAmount = BigDecimal.ZERO;
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

