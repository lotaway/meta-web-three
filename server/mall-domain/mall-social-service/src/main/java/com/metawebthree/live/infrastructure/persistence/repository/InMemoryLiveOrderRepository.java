package com.metawebthree.live.infrastructure.persistence.repository;

import com.metawebthree.live.domain.model.LiveOrder;
import com.metawebthree.live.domain.repository.LiveOrderRepository;
import org.springframework.stereotype.Repository;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

/**
 * In-memory LiveOrderRepository. Live has no persistence layer yet;
 * this keeps the live service startable for development.
 */
@Repository
public class InMemoryLiveOrderRepository implements LiveOrderRepository {

    private final Map<Long, LiveOrder> store = new ConcurrentHashMap<>();
    private final AtomicLong idSeq = new AtomicLong(1);

    @Override
    public LiveOrder save(LiveOrder order) {
        if (order.getId() == null) {
            order.setId(idSeq.getAndIncrement());
        }
        store.put(order.getId(), order);
        return order;
    }

    @Override
    public LiveOrder findById(Long id) {
        return store.get(id);
    }

    @Override
    public LiveOrder findByOrderId(Long orderId) {
        if (orderId == null) {
            return null;
        }
        return store.values().stream()
                .filter(o -> orderId.equals(o.getOrderId()))
                .findFirst().orElse(null);
    }

    @Override
    public List<LiveOrder> findByRoomId(Long roomId) {
        if (roomId == null) {
            return new ArrayList<>();
        }
        return store.values().stream()
                .filter(o -> roomId.equals(o.getRoomId()))
                .collect(Collectors.toList());
    }

    @Override
    public List<LiveOrder> findByUserId(Long userId) {
        if (userId == null) {
            return new ArrayList<>();
        }
        return store.values().stream()
                .filter(o -> userId.equals(o.getUserId()))
                .collect(Collectors.toList());
    }

    @Override
    public List<LiveOrder> findAll() {
        return new ArrayList<>(store.values());
    }

    @Override
    public void deleteById(Long id) {
        store.remove(id);
    }
}