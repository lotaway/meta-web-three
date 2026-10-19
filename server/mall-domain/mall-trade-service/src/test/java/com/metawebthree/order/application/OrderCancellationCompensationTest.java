package com.metawebthree.order.application;

import com.baomidou.mybatisplus.core.conditions.Wrapper;
import com.metawebthree.cart.application.CartService;
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
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Cancelling an order must be idempotent and must compensate side effects
 * (inventory reservation, coupon, points).
 */
class OrderCancellationCompensationTest {

    private OrderMapper orderMapper;
    private InventoryClient inventoryClient;
    private OrderPromotionClient promotionClient;
    private UserClient userClient;
    private OrderApplicationService service;

    @BeforeEach
    void setUp() {
        orderMapper = mock(OrderMapper.class);
        inventoryClient = mock(InventoryClient.class);
        promotionClient = mock(OrderPromotionClient.class);
        userClient = mock(UserClient.class);
        service = new OrderApplicationService(
                orderMapper,
                mock(OrderItemMapper.class),
                mock(CommissionSettlementPort.class),
                mock(OrderEventPublisher.class),
                promotionClient,
                userClient,
                inventoryClient,
                mock(PaymentRefundPort.class),
                mock(CartService.class),
                mock(CouponQueryService.class),
                mock(CouponCommandService.class));
        when(inventoryClient.releaseInventoryByOrderId(any(), any())).thenReturn(true);
        when(promotionClient.returnCoupon(any(), any(), any())).thenReturn(true);
        when(userClient.returnIntegration(any(), anyInt(), any())).thenReturn(100);
    }

    private OrderDO order(long id, long userId, String status) {
        OrderDO order = new OrderDO();
        order.setId(id);
        order.setUserId(userId);
        order.setOrderStatus(status);
        order.setOrderNo("NO" + id);
        return order;
    }

    @Test
    void cancelOrderRejectsUnknownOrder() {
        when(orderMapper.selectById(1L)).thenReturn(null);

        assertFalse(service.cancelOrder(1L, 100L));
        verify(inventoryClient, never()).releaseInventoryByOrderId(any(), any());
    }

    @Test
    void cancelOrderRejectsForeignOrder() {
        when(orderMapper.selectById(1L)).thenReturn(order(1L, 100L, "CREATED"));

        assertFalse(service.cancelOrder(1L, 200L));
        verify(inventoryClient, never()).releaseInventoryByOrderId(any(), any());
    }

    @Test
    void cancelOrderIsIdempotentForAlreadyCanceledOrder() {
        when(orderMapper.selectById(1L)).thenReturn(order(1L, 100L, "CANCELED"));

        assertTrue(service.cancelOrder(1L, 100L));
        verify(orderMapper, never()).updateById(any(OrderDO.class));
        verify(inventoryClient, never()).releaseInventoryByOrderId(any(), any());
    }

    @Test
    void cancelOrderMarksCanceledAndReleasesInventory() {
        when(orderMapper.selectById(1L)).thenReturn(order(1L, 100L, "CREATED"));
        when(orderMapper.updateById(any(OrderDO.class))).thenReturn(1);

        assertTrue(service.cancelOrder(1L, 100L));

        verify(orderMapper).updateById(any(OrderDO.class));
        verify(inventoryClient).releaseInventoryByOrderId(1L, "Order canceled");
    }

    @Test
    void cancelTimeOutOrderCompensatesAllSideEffects() {
        OrderDO withCoupon = order(1L, 100L, "CREATED");
        withCoupon.setCouponId(5L);
        withCoupon.setUseIntegration(10);
        OrderDO plain = order(2L, 100L, "CREATED");
        when(orderMapper.selectList(any(Wrapper.class))).thenReturn(List.of(withCoupon, plain));
        when(orderMapper.updateById(any(OrderDO.class))).thenReturn(1);

        int canceled = service.cancelTimeOutOrder(30);

        assertEquals(2, canceled);
        assertEquals("CANCELED", withCoupon.getOrderStatus());
        assertEquals("CANCELED", plain.getOrderStatus());
        verify(inventoryClient, times(2)).releaseInventoryByOrderId(any(), any());
        verify(promotionClient).returnCoupon(100L, 5L, 1L);
        verify(userClient).returnIntegration(100L, 10, 1L);
        verify(promotionClient, times(1)).returnCoupon(any(), any(), any());
        verify(userClient, times(1)).returnIntegration(any(), anyInt(), any());
    }

    @Test
    void cancelTimeOutOrderDoesNothingWhenNoTimeoutOrders() {
        when(orderMapper.selectList(any(Wrapper.class))).thenReturn(List.of());

        assertEquals(0, service.cancelTimeOutOrder(30));
        verify(orderMapper, never()).updateById(any(OrderDO.class));
        verify(inventoryClient, never()).releaseInventoryByOrderId(any(), any());
    }
}
