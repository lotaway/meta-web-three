package com.metawebthree.order.application;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.lang.reflect.Field;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import com.metawebthree.cart.application.CartService;
import com.metawebthree.cart.dto.CartItemDTO;
import com.metawebthree.common.exception.BusinessException;
import com.metawebthree.common.generated.rpc.GetProductDetailRequest;
import com.metawebthree.common.generated.rpc.GetProductDetailResponse;
import com.metawebthree.common.generated.rpc.ProductDetailProto;
import com.metawebthree.common.generated.rpc.ProductService;
import com.metawebthree.order.domain.model.OrderDO;
import com.metawebthree.order.domain.ports.CommissionSettlementPort;
import com.metawebthree.order.domain.ports.PaymentRefundPort;
import com.metawebthree.order.infrastructure.client.InventoryClient;
import com.metawebthree.order.infrastructure.client.OrderPromotionClient;
import com.metawebthree.order.infrastructure.client.UserClient;
import com.metawebthree.order.infrastructure.persistence.mapper.OrderItemMapper;
import com.metawebthree.order.infrastructure.persistence.mapper.OrderMapper;
import com.metawebthree.promotion.application.CouponCommandService;
import com.metawebthree.promotion.application.CouponQueryService;
import com.metawebthree.promotion.domain.model.Coupon;
import com.metawebthree.promotion.domain.model.CouponType;

/**
 * Verifies that {@link OrderApplicationService#generateConfirmOrder} aggregates
 * real cart/product/address/coupon data and that the confirmed amount equals the
 * amount persisted by {@link OrderApplicationService#createOrder}.
 */
class OrderConfirmOrderTest {

    private static final Long USER_ID = 5L;

    private OrderMapper orderMapper;
    private CartService cartService;
    private CouponQueryService couponQueryService;
    private CouponCommandService couponCommandService;
    private UserClient userClient;
    private InventoryClient inventoryClient;
    private ProductService productService;
    private OrderApplicationService service;

    @BeforeEach
    void setUp() throws Exception {
        orderMapper = mock(OrderMapper.class);
        cartService = mock(CartService.class);
        couponQueryService = mock(CouponQueryService.class);
        couponCommandService = mock(CouponCommandService.class);
        userClient = mock(UserClient.class);
        inventoryClient = mock(InventoryClient.class);
        productService = mock(ProductService.class);

        service = new OrderApplicationService(
                orderMapper,
                mock(OrderItemMapper.class),
                mock(CommissionSettlementPort.class),
                mock(OrderEventPublisher.class),
                mock(OrderPromotionClient.class),
                userClient,
                inventoryClient,
                mock(PaymentRefundPort.class),
                cartService,
                couponQueryService,
                couponCommandService);
        setField(service, "productService", productService);
        setField(service, "pointsPerCurrencyUnit", 100);

        when(cartService.list(anyLong())).thenReturn(new ArrayList<>());
        when(couponQueryService.listByOwner(anyLong(), anyInt())).thenReturn(new ArrayList<>());
        when(userClient.listMemberAddresses(anyLong())).thenReturn(new ArrayList<>());
        when(userClient.getUserIntegration(anyLong())).thenReturn(0);
        when(inventoryClient.getAvailableQuantity(any(), any())).thenReturn(null);
        when(inventoryClient.reserveInventory(any(), any(), any(), any())).thenReturn(true);
        when(orderMapper.insert(any(OrderDO.class))).thenReturn(1);
    }

    private static void setField(Object target, String name, Object value) throws Exception {
        Field field = target.getClass().getDeclaredField(name);
        field.setAccessible(true);
        field.set(target, value);
    }

    private CartItemDTO cartItem(long id, long productId, long skuId, int quantity, String price) {
        return CartItemDTO.builder()
                .id(id)
                .productId(productId)
                .productSkuId(skuId)
                .quantity(quantity)
                .price(new BigDecimal(price))
                .build();
    }

    private void stubProduct(long productId, String name, double price) {
        ProductDetailProto product = ProductDetailProto.newBuilder()
                .setId(productId)
                .setName(name)
                .setPic("pic-" + productId)
                .setPrice(price)
                .build();
        when(productService.getProductDetail(any(GetProductDetailRequest.class)))
                .thenReturn(GetProductDetailResponse.newBuilder().setProduct(product).build());
    }

    private CouponType couponType(long id, String discount, String minimum) {
        CouponType type = new CouponType();
        type.setId(id);
        type.setName("type-" + id);
        type.setIsEnabled(true);
        type.setDiscountAmount(new BigDecimal(discount));
        type.setMinimumOrderAmount(minimum != null ? new BigDecimal(minimum) : null);
        type.setStartTime(LocalDateTime.now().minusDays(1));
        type.setEndTime(LocalDateTime.now().plusDays(1));
        return type;
    }

    private Coupon coupon(long id, long typeId, String code) {
        Coupon coupon = new Coupon();
        coupon.setId(id);
        coupon.setCouponTypeId(typeId);
        coupon.setCode(code);
        coupon.setUseStatus(0);
        return coupon;
    }

    @Test
    void rejectsEmptyCart() {
        when(cartService.list(USER_ID)).thenReturn(new ArrayList<>());
        assertThrows(BusinessException.class, () -> service.generateConfirmOrder(USER_ID, Arrays.asList(1L)));
    }

    @Test
    void usesServerPriceAndFlagsPriceChange() {
        when(cartService.list(USER_ID))
                .thenReturn(Collections.singletonList(cartItem(100L, 1L, 2L, 2, "5.00")));
        stubProduct(1L, "Widget", 7.50);

        OrderApplicationService.ConfirmOrderResult result =
                service.generateConfirmOrder(USER_ID, Collections.singletonList(100L));

        assertEquals(1, result.getCartPromotionItemList().size());
        OrderApplicationService.ConfirmOrderItem item = result.getCartPromotionItemList().get(0);
        assertEquals(100L, item.getCartItemId());
        assertEquals(0, new BigDecimal("7.50").compareTo(item.getUnitPrice()));
        assertEquals(0, new BigDecimal("15.00").compareTo(item.getTotalPrice()));
        assertTrue(item.isPriceChanged(), "stale cart price must be flagged");
        assertEquals(0, new BigDecimal("15.00").compareTo(result.getCalcAmount().getTotalAmount()));
        assertEquals(0, new BigDecimal("15.00").compareTo(result.getCalcAmount().getPayAmount()));
    }

    @Test
    void rejectsInsufficientStock() {
        when(cartService.list(USER_ID))
                .thenReturn(Collections.singletonList(cartItem(100L, 1L, 2L, 2, "7.50")));
        stubProduct(1L, "Widget", 7.50);
        when(inventoryClient.getAvailableQuantity(2L, null)).thenReturn(1);

        assertThrows(BusinessException.class,
                () -> service.generateConfirmOrder(USER_ID, Collections.singletonList(100L)));
    }

    @Test
    void appliesCouponDiscount() {
        when(cartService.list(USER_ID))
                .thenReturn(Collections.singletonList(cartItem(100L, 1L, 2L, 2, "7.50")));
        stubProduct(1L, "Widget", 7.50);
        when(couponQueryService.listByOwner(USER_ID, 0))
                .thenReturn(Collections.singletonList(coupon(9L, 77L, "ABC")));
        when(couponQueryService.getCouponType(77L)).thenReturn(couponType(77L, "3.00", "10.00"));

        OrderApplicationService.ConfirmOrderResult result =
                service.generateConfirmOrder(USER_ID, Collections.singletonList(100L), 9L, null);

        assertEquals(0, new BigDecimal("3.00").compareTo(result.getCalcAmount().getPromotionAmount()));
        assertEquals(0, new BigDecimal("12.00").compareTo(result.getCalcAmount().getPayAmount()));
    }

    @Test
    void rejectsCouponBelowMinimum() {
        when(cartService.list(USER_ID))
                .thenReturn(Collections.singletonList(cartItem(100L, 1L, 2L, 1, "7.50")));
        stubProduct(1L, "Widget", 7.50);
        when(couponQueryService.listByOwner(USER_ID, 0))
                .thenReturn(Collections.singletonList(coupon(9L, 77L, "ABC")));
        when(couponQueryService.getCouponType(77L)).thenReturn(couponType(77L, "3.00", "100.00"));

        assertThrows(BusinessException.class,
                () -> service.generateConfirmOrder(USER_ID, Collections.singletonList(100L), 9L, null));
    }

    @Test
    void appliesPointsDiscount() {
        when(cartService.list(USER_ID))
                .thenReturn(Collections.singletonList(cartItem(100L, 1L, 2L, 2, "7.50")));
        stubProduct(1L, "Widget", 7.50);
        when(userClient.getUserIntegration(USER_ID)).thenReturn(2000);

        OrderApplicationService.ConfirmOrderResult result =
                service.generateConfirmOrder(USER_ID, Collections.singletonList(100L), null, 1000);

        assertEquals(0, new BigDecimal("10.00").compareTo(result.getCalcAmount().getIntegrationAmount()));
        assertEquals(0, new BigDecimal("5.00").compareTo(result.getCalcAmount().getPayAmount()));
    }

    @Test
    void rejectsInsufficientPoints() {
        when(cartService.list(USER_ID))
                .thenReturn(Collections.singletonList(cartItem(100L, 1L, 2L, 2, "7.50")));
        stubProduct(1L, "Widget", 7.50);
        when(userClient.getUserIntegration(USER_ID)).thenReturn(100);

        assertThrows(BusinessException.class,
                () -> service.generateConfirmOrder(USER_ID, Collections.singletonList(100L), null, 1000));
    }

    @Test
    void createOrderAmountMatchesConfirmPreview() {
        when(cartService.list(USER_ID))
                .thenReturn(Collections.singletonList(cartItem(100L, 1L, 2L, 2, "7.50")));
        stubProduct(1L, "Widget", 7.50);
        when(couponQueryService.listByOwner(USER_ID, 0))
                .thenReturn(Collections.singletonList(coupon(9L, 77L, "ABC")));
        when(couponQueryService.getCouponType(77L)).thenReturn(couponType(77L, "3.00", "10.00"));
        when(userClient.getUserIntegration(USER_ID)).thenReturn(2000);

        // subtotal 15.00 - coupon 3.00 - points 10.00 = 2.00
        OrderApplicationService.ConfirmOrderResult preview =
                service.generateConfirmOrder(USER_ID, Collections.singletonList(100L), 9L, 1000);

        OrderApplicationService.OrderItemCreate item = new OrderApplicationService.OrderItemCreate();
        item.setProductId(1L);
        item.setSkuId(2L);
        item.setQuantity(2);

        service.createOrder(USER_ID, "remark", Collections.singletonList(item), null, 9L, 1000);

        ArgumentCaptor<OrderDO> captor = ArgumentCaptor.forClass(OrderDO.class);
        verify(orderMapper).insert(captor.capture());
        OrderDO stored = captor.getValue();
        assertNotNull(stored.getOrderAmount());
        assertEquals(0, preview.getCalcAmount().getPayAmount().compareTo(stored.getOrderAmount()),
                "confirmed amount must equal stored order amount");
        assertEquals(0, new BigDecimal("2.00").compareTo(stored.getOrderAmount()));
    }

    @Test
    void createOrderConsumesAppliedCoupon() {
        when(cartService.list(USER_ID)).thenReturn(new ArrayList<>());
        stubProduct(1L, "Widget", 7.50);
        when(couponQueryService.listByOwner(USER_ID, 0))
                .thenReturn(Collections.singletonList(coupon(9L, 77L, "ABC")));
        when(couponQueryService.getCouponType(77L)).thenReturn(couponType(77L, "3.00", "10.00"));

        OrderApplicationService.OrderItemCreate item = new OrderApplicationService.OrderItemCreate();
        item.setProductId(1L);
        item.setSkuId(2L);
        item.setQuantity(2);

        service.createOrder(USER_ID, "remark", Collections.singletonList(item), null, 9L, null);

        verify(couponCommandService).consume(any(), anyLong(), any(), any(), any());
    }
}
