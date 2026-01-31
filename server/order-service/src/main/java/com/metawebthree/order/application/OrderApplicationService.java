package com.metawebthree.order.application;

import java.math.BigDecimal;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.metawebthree.order.domain.model.OrderDO;
import com.metawebthree.order.domain.model.OrderItemDO;
import com.metawebthree.order.infrastructure.persistence.mapper.OrderItemMapper;
import com.metawebthree.order.infrastructure.persistence.mapper.OrderMapper;

import lombok.Data;

@Service
public class OrderApplicationService {
    private final OrderMapper orderMapper;
    private final OrderItemMapper orderItemMapper;

    public OrderApplicationService(OrderMapper orderMapper, OrderItemMapper orderItemMapper) {
        this.orderMapper = orderMapper;
        this.orderItemMapper = orderItemMapper;
    }

    @Transactional
    public Long createOrder(Long userId, String remark, List<OrderItemCreate> items) {
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

    @Data
    public static class OrderItemCreate {
        private Long productId;
        private String productName;
        private Long skuId;
        private Integer quantity;
        private BigDecimal unitPrice;
        private String imageUrl;
    }

    @Data
    public static class OrderCreateRequest {
        private String remark;
        private List<OrderItemCreate> items;
    }

    @Data
    public static class OrderWithItems {
        private OrderDO order;
        private List<OrderItemDO> items;
        public BigDecimal getTotalPrice() {
            return items.stream().map(OrderItemDO::getTotalPrice).reduce(BigDecimal.ZERO, BigDecimal::add);
        }
    }
}

