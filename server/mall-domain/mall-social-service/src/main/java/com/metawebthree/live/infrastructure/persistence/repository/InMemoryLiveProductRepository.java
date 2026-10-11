package com.metawebthree.live.infrastructure.persistence.repository;

import com.metawebthree.live.domain.model.LiveProduct;
import com.metawebthree.live.domain.repository.LiveProductRepository;
import org.springframework.stereotype.Repository;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

/**
 * In-memory LiveProductRepository. Live has no persistence layer yet;
 * this keeps the live service startable for development.
 */
@Repository
public class InMemoryLiveProductRepository implements LiveProductRepository {

    private final Map<Long, LiveProduct> store = new ConcurrentHashMap<>();
    private final AtomicLong idSeq = new AtomicLong(1);

    @Override
    public LiveProduct save(LiveProduct product) {
        if (product.getId() == null) {
            product.setId(idSeq.getAndIncrement());
        }
        store.put(product.getId(), product);
        return product;
    }

    @Override
    public LiveProduct findById(Long id) {
        return store.get(id);
    }

    @Override
    public List<LiveProduct> findByRoomId(Long roomId) {
        if (roomId == null) {
            return new ArrayList<>();
        }
        return store.values().stream()
                .filter(p -> roomId.equals(p.getRoomId()))
                .collect(Collectors.toList());
    }

    @Override
    public List<LiveProduct> findByProductId(Long productId) {
        if (productId == null) {
            return new ArrayList<>();
        }
        return store.values().stream()
                .filter(p -> productId.equals(p.getProductId()))
                .collect(Collectors.toList());
    }

    @Override
    public List<LiveProduct> findAll() {
        return new ArrayList<>(store.values());
    }

    @Override
    public void deleteById(Long id) {
        store.remove(id);
    }
}