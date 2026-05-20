package com.metawebthree.order.application;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

import com.metawebthree.order.domain.model.OrderReturnApply;
import com.metawebthree.order.domain.repository.OrderReturnRepository;

class OrderReturnApplicationServiceTest {

    private final OrderReturnRepository repository = new InMemoryOrderReturnRepository();
    private final OrderReturnApplicationService service = new OrderReturnApplicationService(repository);

    @Test
    void applyReturnAcceptsOrderNoAndReturnName() {
        OrderReturnApply apply = OrderReturnApply.builder()
                .id(1L)
                .orderNo("ORDER-1001")
                .returnName("tester")
                .build();

        assertDoesNotThrow(() -> service.applyReturn(apply));
    }

    @Test
    void applyReturnRejectsMissingOrderIdentity() {
        OrderReturnApply apply = OrderReturnApply.builder()
                .id(1L)
                .returnName("tester")
                .build();

        assertThrows(IllegalArgumentException.class, () -> service.applyReturn(apply));
    }

    private static final class InMemoryOrderReturnRepository implements OrderReturnRepository {

        @Override
        public void save(OrderReturnApply apply) {
        }

        @Override
        public void update(OrderReturnApply apply) {
        }

        @Override
        public OrderReturnApply findById(Long id) {
            return null;
        }

        @Override
        public java.util.List<OrderReturnApply> findByOrderSn(String orderSn) {
            return java.util.List.of();
        }

        @Override
        public java.util.List<OrderReturnApply> findByStatus(Integer status) {
            return java.util.List.of();
        }

        @Override
        public void delete(Long id) {
        }
    }
}
