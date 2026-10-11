package com.metawebthree.order.application;

import com.metawebthree.order.domain.model.OrderReturnApply;
import com.metawebthree.order.domain.repository.OrderReturnRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import java.util.List;

@Service
@RequiredArgsConstructor
public class OrderReturnApplicationService {

    private final OrderReturnRepository repository;

    public void applyReturn(OrderReturnApply apply) {
        validateApply(apply);
        repository.save(apply);
    }

    public void handleReturn(OrderReturnApply apply) {
        validateId(apply.getId());
        repository.update(apply);
    }

    public OrderReturnApply getReturn(Long id) {
        validateId(id);
        return repository.findById(id);
    }

    public List<OrderReturnApply> listByOrder(String orderSn) {
        return repository.findByOrderSn(orderSn);
    }

    public void removeHistory(Long id) {
        validateId(id);
        repository.delete(id);
    }

    private void validateApply(OrderReturnApply apply) {
        if (apply.getOrderNo() == null || apply.getReturnName() == null) {
            throw new IllegalArgumentException("Order identity and returner info are required");
        }
    }

    private void validateId(Long id) {
        if (id == null) {
            throw new IllegalArgumentException("Invalid return application identity");
        }
    }
}
