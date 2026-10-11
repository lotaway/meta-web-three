package com.metawebthree.live.infrastructure.persistence.repository;

import com.metawebthree.live.domain.model.LiveComment;
import com.metawebthree.live.domain.repository.LiveCommentRepository;
import org.springframework.stereotype.Repository;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

/**
 * In-memory LiveCommentRepository. Live has no persistence layer yet;
 * this keeps the live service startable for development.
 */
@Repository
public class InMemoryLiveCommentRepository implements LiveCommentRepository {

    private final Map<Long, LiveComment> store = new ConcurrentHashMap<>();
    private final AtomicLong idSeq = new AtomicLong(1);

    @Override
    public LiveComment save(LiveComment comment) {
        if (comment.getId() == null) {
            comment.setId(idSeq.getAndIncrement());
        }
        store.put(comment.getId(), comment);
        return comment;
    }

    @Override
    public LiveComment findById(Long id) {
        return store.get(id);
    }

    @Override
    public List<LiveComment> findByRoomId(Long roomId) {
        if (roomId == null) {
            return new ArrayList<>();
        }
        return store.values().stream()
                .filter(c -> roomId.equals(c.getRoomId()))
                .collect(Collectors.toList());
    }

    @Override
    public List<LiveComment> findByUserId(Long userId) {
        if (userId == null) {
            return new ArrayList<>();
        }
        return store.values().stream()
                .filter(c -> userId.equals(c.getUserId()))
                .collect(Collectors.toList());
    }

    @Override
    public List<LiveComment> findAll() {
        return new ArrayList<>(store.values());
    }

    @Override
    public void deleteById(Long id) {
        store.remove(id);
    }
}