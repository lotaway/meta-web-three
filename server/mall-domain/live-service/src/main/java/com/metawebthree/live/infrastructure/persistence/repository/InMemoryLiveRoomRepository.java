package com.metawebthree.live.infrastructure.persistence.repository;

import com.metawebthree.live.domain.model.LiveRoom;
import com.metawebthree.live.domain.repository.LiveRoomRepository;
import org.springframework.stereotype.Repository;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

/**
 * In-memory LiveRoomRepository. Live has no persistence layer yet;
 * this keeps the live service startable for development.
 */
@Repository
public class InMemoryLiveRoomRepository implements LiveRoomRepository {

    private final Map<Long, LiveRoom> store = new ConcurrentHashMap<>();
    private final AtomicLong idSeq = new AtomicLong(1);

    @Override
    public LiveRoom save(LiveRoom room) {
        if (room.getId() == null) {
            room.setId(idSeq.getAndIncrement());
        }
        store.put(room.getId(), room);
        return room;
    }

    @Override
    public LiveRoom findById(Long id) {
        return store.get(id);
    }

    @Override
    public List<LiveRoom> findByAnchorId(Long anchorId) {
        if (anchorId == null) {
            return new ArrayList<>();
        }
        return store.values().stream()
                .filter(r -> anchorId.equals(r.getAnchorId()))
                .collect(Collectors.toList());
    }

    @Override
    public List<LiveRoom> findByStatus(Integer status) {
        if (status == null) {
            return new ArrayList<>();
        }
        return store.values().stream()
                .filter(r -> status.equals(r.getStatus()))
                .collect(Collectors.toList());
    }

    @Override
    public List<LiveRoom> findAll() {
        return new ArrayList<>(store.values());
    }

    @Override
    public void deleteById(Long id) {
        store.remove(id);
    }
}