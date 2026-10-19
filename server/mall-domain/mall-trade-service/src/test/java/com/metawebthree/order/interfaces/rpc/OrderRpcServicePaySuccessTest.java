package com.metawebthree.order.interfaces.rpc;

import com.baomidou.mybatisplus.core.MybatisConfiguration;
import com.baomidou.mybatisplus.core.metadata.TableInfoHelper;
import com.metawebthree.common.generated.rpc.PaySuccessRequest;
import com.metawebthree.common.generated.rpc.PaySuccessResponse;
import com.metawebthree.order.application.AdminOrderQueryService;
import com.metawebthree.order.domain.model.OrderDO;
import com.metawebthree.order.infrastructure.client.InventoryClient;
import com.metawebthree.order.infrastructure.persistence.mapper.OrderItemMapper;
import com.metawebthree.order.infrastructure.persistence.mapper.OrderMapper;
import com.metawebthree.order.infrastructure.persistence.mapper.ReturnApplyMapper;
import org.apache.ibatis.builder.MapperBuilderAssistant;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class OrderRpcServicePaySuccessTest {

    private OrderMapper orderMapper;
    private OrderRpcService service;

    @BeforeAll
    static void initMybatis() {
        MapperBuilderAssistant assistant = new MapperBuilderAssistant(
                new com.baomidou.mybatisplus.core.MybatisConfiguration(), "");
        com.baomidou.mybatisplus.core.metadata.TableInfoHelper.initTableInfo(assistant, OrderDO.class);
    }

    @BeforeEach
    void setUp() {
        orderMapper = mock(OrderMapper.class);
        service = new OrderRpcService(
                mock(AdminOrderQueryService.class),
                orderMapper,
                mock(OrderItemMapper.class),
                mock(ReturnApplyMapper.class),
                mock(InventoryClient.class));
    }

    private PaySuccessRequest payRequest(Long orderId, String transactionId, String channel, String amount) {
        PaySuccessRequest.Builder builder = PaySuccessRequest.newBuilder()
                .setOrderId(orderId)
                .setPayType(2);
        if (transactionId != null) {
            builder.setTransactionId(transactionId);
        }
        if (channel != null) {
            builder.setChannel(channel);
        }
        if (amount != null) {
            builder.setTotalAmount(amount);
        }
        return builder.build();
    }

    @Test
    void createdOrderBecomesPaid() {
        OrderDO order = OrderDO.builder().id(1L).orderStatus("CREATED")
                .orderAmount(new BigDecimal("10.00")).build();
        when(orderMapper.selectById(1L)).thenReturn(order);
        when(orderMapper.update(isNull(), any())).thenReturn(1);

        PaySuccessResponse response = service.paySuccess(payRequest(1L, "TX-1", "ALIPAY", "10.00"));

        assertTrue(response.getSuccess());
        verify(orderMapper).update(isNull(), any());
    }

    @Test
    void alreadyPaidOrderIsIdempotent() {
        OrderDO order = OrderDO.builder().id(2L).orderStatus("PAID").build();
        when(orderMapper.selectById(2L)).thenReturn(order);

        PaySuccessResponse response = service.paySuccess(payRequest(2L, "TX-2", "ALIPAY", null));

        assertTrue(response.getSuccess());
        verify(orderMapper, never()).update(isNull(), any());
    }

    @Test
    void missingTransactionIdIsRejected() {
        PaySuccessResponse response = service.paySuccess(payRequest(3L, null, "ALIPAY", null));
        assertFalse(response.getSuccess());
    }

    @Test
    void missingChannelIsRejected() {
        PaySuccessResponse response = service.paySuccess(payRequest(4L, "TX-4", null, null));
        assertFalse(response.getSuccess());
    }

    @Test
    void amountMismatchIsRejected() {
        OrderDO order = OrderDO.builder().id(5L).orderStatus("CREATED")
                .orderAmount(new BigDecimal("10.00")).build();
        when(orderMapper.selectById(5L)).thenReturn(order);

        PaySuccessResponse response = service.paySuccess(payRequest(5L, "TX-5", "ALIPAY", "9.00"));

        assertFalse(response.getSuccess());
        verify(orderMapper, never()).update(isNull(), any());
    }

    @Test
    void illegalStatusTransitionIsRejected() {
        OrderDO order = OrderDO.builder().id(6L).orderStatus("CANCELED").build();
        when(orderMapper.selectById(6L)).thenReturn(order);

        PaySuccessResponse response = service.paySuccess(payRequest(6L, "TX-6", "ALIPAY", null));

        assertFalse(response.getSuccess());
    }

    @Test
    void missingOrderIsRejected() {
        when(orderMapper.selectById(7L)).thenReturn(null);

        PaySuccessResponse response = service.paySuccess(payRequest(7L, "TX-7", "ALIPAY", null));

        assertFalse(response.getSuccess());
    }

    @Test
    void zeroUpdatedFallsBackToRequery() {
        OrderDO order = OrderDO.builder().id(8L).orderStatus("CREATED")
                .orderAmount(new BigDecimal("10.00")).build();
        when(orderMapper.selectById(8L)).thenReturn(order);
        when(orderMapper.update(isNull(), any())).thenReturn(0);

        PaySuccessResponse response = service.paySuccess(payRequest(8L, "TX-8", "ALIPAY", null));

        assertFalse(response.getSuccess());
    }
}
