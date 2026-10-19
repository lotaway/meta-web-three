package com.metawebthree.aftersale.application.service;

import com.baomidou.mybatisplus.core.metadata.TableInfoHelper;
import com.metawebthree.aftersale.application.dto.AfterSaleApplyDTO;
import com.metawebthree.aftersale.domain.model.AfterSaleOrderDO;
import com.metawebthree.aftersale.domain.model.AfterSaleStatus;
import com.metawebthree.aftersale.domain.repository.AfterSaleRepository;
import com.metawebthree.aftersale.infrastructure.client.OrderClient;
import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.common.exception.BusinessException;
import com.metawebthree.order.domain.model.OrderDO;
import com.metawebthree.order.domain.model.OrderItemDO;
import com.metawebthree.order.infrastructure.persistence.mapper.OrderItemMapper;
import com.metawebthree.order.infrastructure.persistence.mapper.OrderMapper;
import org.apache.ibatis.builder.MapperBuilderAssistant;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class AfterSaleApplicationServiceTest {

    private AfterSaleRepository afterSaleRepository;
    private OrderMapper orderMapper;
    private OrderItemMapper orderItemMapper;
    private AfterSaleApplicationService service;

    @BeforeAll
    static void initMybatis() {
        MapperBuilderAssistant assistant = new MapperBuilderAssistant(
                new com.baomidou.mybatisplus.core.MybatisConfiguration(), "");
        TableInfoHelper.initTableInfo(assistant, OrderItemDO.class);
    }

    @BeforeEach
    void setUp() {
        afterSaleRepository = mock(AfterSaleRepository.class);
        orderMapper = mock(OrderMapper.class);
        orderItemMapper = mock(OrderItemMapper.class);
        service = new AfterSaleApplicationService(afterSaleRepository, mock(OrderClient.class),
                orderMapper, orderItemMapper);
        when(afterSaleRepository.findByOrderId(anyLong())).thenReturn(new ArrayList<>());
    }

    private OrderDO paidOrder() {
        return OrderDO.builder().id(100L).userId(5L).orderNo("NO100")
                .orderStatus("PAID").orderAmount(new BigDecimal("30.00")).build();
    }

    private OrderItemDO orderItem() {
        return OrderItemDO.builder().id(1L).orderId(100L).productId(7L).skuId(9L)
                .productName("Widget").imageUrl("img.png").quantity(3)
                .unitPrice(new BigDecimal("10.00")).build();
    }

    private AfterSaleApplyDTO applyDTO() {
        AfterSaleApplyDTO dto = new AfterSaleApplyDTO();
        dto.setOrderId(100L);
        dto.setProductId(7L);
        dto.setSkuId(9L);
        dto.setQuantity(2);
        dto.setRefundAmount(2000);
        dto.setAfterSaleType(3);
        dto.setApplyReason("broken");
        return dto;
    }

    private BusinessException expectRejected(Runnable action) {
        return assertThrows(BusinessException.class, action::run);
    }

    @Test
    void orderNotFoundIsRejected() {
        when(orderMapper.selectById(100L)).thenReturn(null);
        expectRejected(() -> service.apply(applyDTO(), 5L));
    }

    @Test
    void otherUsersOrderIsRejected() {
        when(orderMapper.selectById(100L)).thenReturn(paidOrder());
        BusinessException ex = expectRejected(() -> service.apply(applyDTO(), 999L));
        assertEquals(ResponseStatus.FORBIDDEN, ex.getStatus());
    }

    @Test
    void unpaidOrderIsRejected() {
        OrderDO order = paidOrder();
        order.setOrderStatus("CREATED");
        when(orderMapper.selectById(100L)).thenReturn(order);
        BusinessException ex = expectRejected(() -> service.apply(applyDTO(), 5L));
        assertEquals(ResponseStatus.ORDER_STATUS_INVALID, ex.getStatus());
    }

    @Test
    void skuNotBelongingToOrderIsRejected() {
        when(orderMapper.selectById(100L)).thenReturn(paidOrder());
        when(orderItemMapper.selectList(any())).thenReturn(new ArrayList<>());
        expectRejected(() -> service.apply(applyDTO(), 5L));
    }

    @Test
    void quantityBeyondOrderedIsRejected() {
        when(orderMapper.selectById(100L)).thenReturn(paidOrder());
        when(orderItemMapper.selectList(any())).thenReturn(List.of(orderItem()));
        AfterSaleApplyDTO dto = applyDTO();
        dto.setQuantity(4);
        expectRejected(() -> service.apply(dto, 5L));
    }

    @Test
    void duplicateAfterSaleBeyondOrderedIsRejected() {
        when(orderMapper.selectById(100L)).thenReturn(paidOrder());
        when(orderItemMapper.selectList(any())).thenReturn(List.of(orderItem()));
        AfterSaleOrderDO existing = new AfterSaleOrderDO();
        existing.setOrderId(100L);
        existing.setSkuId(9L);
        existing.setQuantity(2);
        existing.setAfterSaleStatus(AfterSaleStatus.PENDING.getCode());
        when(afterSaleRepository.findByOrderId(100L)).thenReturn(List.of(existing));
        expectRejected(() -> service.apply(applyDTO(), 5L));
    }

    @Test
    void refundAmountBeyondLineTotalIsRejected() {
        when(orderMapper.selectById(100L)).thenReturn(paidOrder());
        when(orderItemMapper.selectList(any())).thenReturn(List.of(orderItem()));
        AfterSaleApplyDTO dto = applyDTO();
        dto.setQuantity(3);
        dto.setRefundAmount(3001);
        expectRejected(() -> service.apply(dto, 5L));
    }

    @Test
    void validApplyPersistsOrderItemSnapshot() {
        when(orderMapper.selectById(100L)).thenReturn(paidOrder());
        when(orderItemMapper.selectList(any())).thenReturn(List.of(orderItem()));

        service.apply(applyDTO(), 5L);

        ArgumentCaptor<AfterSaleOrderDO> captor = ArgumentCaptor.forClass(AfterSaleOrderDO.class);
        verify(afterSaleRepository).save(captor.capture());
        AfterSaleOrderDO saved = captor.getValue();
        assertEquals(100L, saved.getOrderId());
        assertEquals(5L, saved.getUserId());
        assertEquals(7L, saved.getProductId());
        assertEquals(9L, saved.getSkuId());
        assertEquals("Widget", saved.getProductName());
        assertEquals("img.png", saved.getProductImage());
        assertEquals(2, saved.getQuantity());
        assertEquals(2000, saved.getRefundAmount());
        assertEquals(AfterSaleStatus.PENDING.getCode(), saved.getAfterSaleStatus());
        assertNotNull(saved.getApplyTime());
    }
}
