package com.metawebthree.order.application;

import com.metawebthree.order.domain.model.OrderDO;
import com.metawebthree.order.domain.ports.CommissionSettlementPort;
import com.metawebthree.order.domain.ports.PaymentRefundPort;
import com.metawebthree.order.infrastructure.client.InventoryClient;
import com.metawebthree.order.infrastructure.client.OrderPromotionClient;
import com.metawebthree.order.infrastructure.client.UserClient;
import com.metawebthree.order.infrastructure.persistence.mapper.OrderItemMapper;
import com.metawebthree.order.infrastructure.persistence.mapper.OrderMapper;
import org.apache.ibatis.builder.MapperBuilderAssistant;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class OrderApplicationServiceRefundTest {

    private OrderMapper orderMapper;
    private PaymentRefundPort paymentRefundPort;
    private OrderApplicationService service;

    @BeforeAll
    static void initMybatis() {
        MapperBuilderAssistant assistant = new MapperBuilderAssistant(
                new com.baomidou.mybatisplus.core.MybatisConfiguration(), "");
        com.baomidou.mybatisplus.core.metadata.TableInfoHelper.initTableInfo(assistant, OrderDO.class);
    }

    @BeforeEach
    void setUp() {
        orderMapper = mock(OrderMapper.class);
        paymentRefundPort = mock(PaymentRefundPort.class);
        service = new OrderApplicationService(
                orderMapper,
                mock(OrderItemMapper.class),
                mock(CommissionSettlementPort.class),
                mock(OrderEventPublisher.class),
                mock(OrderPromotionClient.class),
                mock(UserClient.class),
                mock(InventoryClient.class),
                paymentRefundPort);
    }

    private OrderDO order(String status) {
        return OrderDO.builder().id(1L).userId(5L).orderNo("NO1")
                .orderStatus(status).orderAmount(new BigDecimal("10.00")).build();
    }

    @Test
    void missingOrderIsRejected() {
        when(orderMapper.selectById(1L)).thenReturn(null);
        assertFalse(service.refundOrder(1L, 5L));
        verify(paymentRefundPort, never()).refund(any(), any(), any(), any());
    }

    @Test
    void wrongOwnerIsRejected() {
        when(orderMapper.selectById(1L)).thenReturn(order("PAID"));
        assertFalse(service.refundOrder(1L, 999L));
        verify(paymentRefundPort, never()).refund(any(), any(), any(), any());
    }

    @Test
    void unpaidOrderIsRejected() {
        when(orderMapper.selectById(1L)).thenReturn(order("CREATED"));
        assertFalse(service.refundOrder(1L, 5L));
        verify(paymentRefundPort, never()).refund(any(), any(), any(), any());
        verify(orderMapper, never()).update(isNull(), any());
    }

    @Test
    void alreadyRefundedIsIdempotent() {
        when(orderMapper.selectById(1L)).thenReturn(order("REFUNDED"));
        assertTrue(service.refundOrder(1L, 5L));
        verify(paymentRefundPort, never()).refund(any(), any(), any(), any());
    }

    @Test
    void channelSuccessMarksOrderRefunded() {
        when(orderMapper.selectById(1L)).thenReturn(order("PAID"));
        when(orderMapper.update(isNull(), any())).thenReturn(1);
        when(paymentRefundPort.refund(any(), any(), any(), any())).thenReturn(true);

        assertTrue(service.refundOrder(1L, 5L));

        verify(paymentRefundPort).refund(any(), any(), any(), any());
        verify(orderMapper, times(2)).update(isNull(), any());
    }

    @Test
    void channelFailureRestoresOrder() {
        when(orderMapper.selectById(1L)).thenReturn(order("PAID"));
        when(orderMapper.update(isNull(), any())).thenReturn(1);
        when(paymentRefundPort.refund(any(), any(), any(), any())).thenReturn(false);

        assertFalse(service.refundOrder(1L, 5L));

        verify(paymentRefundPort).refund(any(), any(), any(), any());
        verify(orderMapper, times(2)).update(isNull(), any());
    }

    @Test
    void channelExceptionRestoresOrderAndRethrows() {
        when(orderMapper.selectById(1L)).thenReturn(order("PAID"));
        when(orderMapper.update(isNull(), any())).thenReturn(1);
        when(paymentRefundPort.refund(any(), any(), any(), any()))
                .thenThrow(new RuntimeException("channel down"));

        assertThrows(RuntimeException.class, () -> service.refundOrder(1L, 5L));

        verify(orderMapper, times(2)).update(isNull(), any());
    }

    @Test
    void concurrentRefundClaimLosingCallerReturnsFalseWhenNotRefundedYet() {
        when(orderMapper.selectById(1L)).thenReturn(order("PAID"));
        when(orderMapper.update(isNull(), any())).thenReturn(0);
        // Second select reveals the claim was taken by another caller but not yet finalised.
        when(orderMapper.selectById(1L)).thenReturn(order("PAID"), order("REFUNDING"));

        assertFalse(service.refundOrder(1L, 5L));
        verify(paymentRefundPort, never()).refund(any(), any(), any(), any());
    }
}
