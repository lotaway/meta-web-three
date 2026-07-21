package com.metawebthree.aiwarehouse.infrastructure.persistence.repository;

import com.metawebthree.aiwarehouse.domain.entity.AIRequestRecord;
import com.metawebthree.aiwarehouse.domain.repository.AIRequestRecordRepository;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

@Repository
public class AIRequestRecordRepositoryImpl implements AIRequestRecordRepository {

    private final ConcurrentHashMap<Long, AIRequestRecord> store = new ConcurrentHashMap<>();
    private final AtomicLong idGen = new AtomicLong(1);

    @Override
    public AIRequestRecord save(AIRequestRecord record) {
        if (record.getId() == null) {
            record.setId(idGen.getAndIncrement());
        }
        store.put(record.getId(), record);
        return record;
    }

    @Override
    public Optional<AIRequestRecord> findById(Long id) {
        return Optional.ofNullable(store.get(id));
    }

    @Override
    public Optional<AIRequestRecord> findByRequestId(String requestId) {
        return store.values().stream()
                .filter(r -> requestId.equals(r.getRequestId()))
                .findFirst();
    }

    @Override
    public List<AIRequestRecord> findByCapabilityId(String capabilityId) {
        return store.values().stream()
                .filter(r -> capabilityId.equals(r.getCapabilityId()))
                .collect(Collectors.toList());
    }

    @Override
    public List<AIRequestRecord> findByCallerServiceName(String serviceName) {
        return store.values().stream()
                .filter(r -> serviceName.equals(r.getCallerServiceName()))
                .collect(Collectors.toList());
    }

    @Override
    public List<AIRequestRecord> findByStatus(AIRequestRecord.AIRequestStatus status) {
        return store.values().stream()
                .filter(r -> r.getStatus() == status)
                .collect(Collectors.toList());
    }

    @Override
    public List<AIRequestRecord> findByCreatedAtBetween(LocalDateTime start, LocalDateTime end) {
        return store.values().stream()
                .filter(r -> r.getCreatedAt() != null
                        && !r.getCreatedAt().isBefore(start)
                        && !r.getCreatedAt().isAfter(end))
                .collect(Collectors.toList());
    }

    @Override
    public List<AIRequestRecord> findTop100ByOrderByCreatedAtDesc() {
        return store.values().stream()
                .sorted(Comparator.comparing(AIRequestRecord::getCreatedAt,
                        Comparator.nullsFirst(Comparator.reverseOrder())))
                .limit(100)
                .collect(Collectors.toList());
    }

    @Override
    public void deleteById(Long id) {
        store.remove(id);
    }

    @Override
    public long count() {
        return store.size();
    }
}
