package com.metawebthree.live.infrastructure.persistence.repository;

import com.metawebthree.live.domain.model.Anchor;
import com.metawebthree.live.domain.repository.AnchorRepository;
import org.springframework.stereotype.Repository;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * In-memory AnchorRepository. Live has no persistence layer yet;
 * this keeps the live service startable for development.
 */
@Repository
public class InMemoryAnchorRepository implements AnchorRepository {

    private final Map<Long, Anchor> store = new ConcurrentHashMap<>();
    private final AtomicLong idSeq = new AtomicLong(1);

    @Override
    public Anchor save(Anchor anchor) {
        if (anchor.getId() == null) {
            anchor.setId(idSeq.getAndIncrement());
        }
        store.put(anchor.getId(), anchor);
        return anchor;
    }

    @Override
    public Anchor findById(Long id) {
        return store.get(id);
    }

    @Override
    public Anchor findByUserId(Long userId) {
        if (userId == null) {
            return null;
        }
        return store.values().stream()
                .filter(a -> userId.equals(a.getUserId()))
                .findFirst().orElse(null);
    }

    @Override
    public List<Anchor> findAll() {
        return new ArrayList<>(store.values());
    }

    @Override
    public void deleteById(Long id) {
        store.remove(id);
    }
}