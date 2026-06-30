package com.metawebthree.order.application;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.metawebthree.common.generated.rpc.*;
import com.metawebthree.order.domain.model.OrderDO;
import com.metawebthree.order.domain.model.OrderItemDO;
import com.metawebthree.order.infrastructure.persistence.mapper.OrderItemMapper;
import com.metawebthree.order.infrastructure.persistence.mapper.OrderMapper;

import java.math.BigDecimal;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Captor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
class OrderServiceImplTest {

    @Mock
    private OrderMapper orderMapper;
    @Mock
    private OrderItemMapper orderItemMapper;
    @Mock
    private OrderApplicationService orderApplicationService;

    @InjectMocks
    private OrderServiceImpl orderService;

    @Captor
    private ArgumentCaptor<OrderDO> orderCaptor;
    @Captor
    private ArgumentCaptor<OrderItemDO> orderItemCaptor;

    @Test
    void getOrderByOrderNo_whenExists_shouldReturnOrder() {
        String orderNo = "ORDER-123";
        OrderDO order = OrderDO.builder()
                .id(1L)
                .userId(100L)
                .orderNo(orderNo)
                .orderStatus("CREATED")
                .orderType("NORMAL")
                .orderAmount(BigDecimal.valueOf(99.99))
                .build();

        when(orderMapper.selectOne(any())).thenReturn(order);

        GetOrderByOrderNoRequest request = GetOrderByOrderNoRequest.newBuilder()
                .setOrderNo(orderNo).build();

        GetOrderByOrderNoResponse response = orderService.getOrderByOrderNo(request);

        assertTrue(response.hasOrder());
        assertEquals(orderNo, response.getOrder().getOrderNo());
        assertEquals(100L, response.getOrder().getUserId());
    }

    @Test
    void getOrderByOrderNo_whenNotExists_shouldReturnEmpty() {
        when(orderMapper.selectOne(any())).thenReturn(null);

        GetOrderByOrderNoRequest request = GetOrderByOrderNoRequest.newBuilder()
                .setOrderNo("NONEXISTENT").build();

        GetOrderByOrderNoResponse response = orderService.getOrderByOrderNo(request);

        assertFalse(response.hasOrder());
    }

    @Test
    void listOrders_shouldReturnPaginatedResults() {
        int page = 1;
        int size = 10;

        OrderDO order = OrderDO.builder()
                .id(1L)
                .userId(100L)
                .orderNo("ORDER-1")
                .orderStatus("CREATED")
                .orderType("NORMAL")
                .orderAmount(BigDecimal.valueOf(50.0))
                .build();

        Page<OrderDO> pageResult = new Page<>(page, size, 1);
        pageResult.setRecords(List.of(order));

        when(orderMapper.selectPage(any(Page.class), any(LambdaQueryWrapper.class)))
            .thenReturn(pageResult);

        ListOrdersRequest request = ListOrdersRequest.newBuilder()
                .setPage(page).setSize(size).build();

        ListOrdersResponse response = orderService.listOrders(request);

        assertEquals(page, response.getPage());
        assertEquals(size, response.getSize());
        assertEquals(1, response.getTotal());
        assertEquals(1, response.getOrdersCount());
        assertEquals("ORDER-1", response.getOrders(0).getOrderNo());
    }

    @Test
    void createOrder_shouldCreateSuccessfully() {
        long userId = 1L;

        CreateOrderRequest request = CreateOrderRequest.newBuilder()
                .setUserId(userId)
                .addItems(OrderItemProto.newBuilder()
                        .setProductId(100L)
                        .setProductName("Test Product")
                        .setQuantity(2)
                        .setPrice(5000)
                        .build())
                .addItems(OrderItemProto.newBuilder()
                        .setProductId(101L)
                        .setProductName("Another Product")
                        .setQuantity(1)
                        .setPrice(3000)
                        .build())
                .setOrderRemark("Test order")
                .build();

        when(orderMapper.insert(any(OrderDO.class))).thenReturn(1);
        when(orderItemMapper.insert(any(OrderItemDO.class))).thenReturn(1);

        CreateOrderResponse response = orderService.createOrder(request);

        assertTrue(response.getSuccess());
        assertEquals("Order created successfully", response.getMessage());
        assertNotNull(response.getOrderId());
        assertNotNull(response.getOrderNo());

        verify(orderMapper).insert(orderCaptor.capture());
        OrderDO savedOrder = orderCaptor.getValue();
        assertEquals(userId, savedOrder.getUserId());
        assertEquals("CREATED", savedOrder.getOrderStatus());
        assertEquals("NORMAL", savedOrder.getOrderType());
        assertEquals("Test order", savedOrder.getOrderRemark());

        verify(orderItemMapper, times(2)).insert(orderItemCaptor.capture());
        List<OrderItemDO> savedItems = orderItemCaptor.getAllValues();
        assertEquals(2, savedItems.size());
        assertEquals(100L, savedItems.get(0).getProductId());
        assertEquals("Test Product", savedItems.get(0).getProductName());
        assertEquals(2, savedItems.get(0).getQuantity());
        assertEquals(101L, savedItems.get(1).getProductId());
        assertEquals(1, savedItems.get(1).getQuantity());
    }

    @Test
    void createOrder_withNullRequest_shouldFail() {
        CreateOrderResponse response = orderService.createOrder(null);

        assertFalse(response.getSuccess());
        assertEquals("Request must not be null", response.getMessage());
        verify(orderMapper, never()).insert(any(OrderDO.class));
    }

    @Test
    void createOrder_withEmptyItems_shouldFail() {
        CreateOrderRequest request = CreateOrderRequest.newBuilder()
                .setUserId(1L)
                .build();

        CreateOrderResponse response = orderService.createOrder(request);

        assertFalse(response.getSuccess());
        assertEquals("Order must have at least one item", response.getMessage());
        verify(orderMapper, never()).insert(any(OrderDO.class));
}
}
