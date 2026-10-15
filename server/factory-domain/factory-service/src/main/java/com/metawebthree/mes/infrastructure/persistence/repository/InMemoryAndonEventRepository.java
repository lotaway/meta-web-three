package com.metawebthree.mes.infrastructure.persistence.repository;

import com.metawebthree.mes.domain.entity.AndonEvent;
import com.metawebthree.mes.domain.repository.AndonEventRepository;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Predicate;
import java.util.stream.Collectors;

/**
 * Opt-in, non-durable event storage for isolated development only.
 * Default profiles use PostgreSQL through AndonEventRepositoryImpl.
 */
@Repository
@org.springframework.context.annotation.Profile("mes-in-memory")
public class InMemoryAndonEventRepository implements AndonEventRepository {

    private final Map<Long, AndonEvent> store = new ConcurrentHashMap<>();
    private final AtomicLong idSeq = new AtomicLong(1);

    @Override
    public Optional<AndonEvent> findById(Long id) {
        return Optional.ofNullable(store.get(id));
    }

    @Override
    public Optional<AndonEvent> findByEventNo(String eventNo) {
        if (eventNo == null) {
            return Optional.empty();
        }
        return findWhere(event -> eventNo.equals(event.getEventNo())).stream().findFirst();
    }

    @Override
    public List<AndonEvent> findByStatus(AndonEvent.AndonEventStatus status) {
        return findWhere(event -> event.getStatus() == status);
    }

    @Override
    public List<AndonEvent> findByAndonTypeId(Long andonTypeId) {
        return findWhere(event -> matches(andonTypeId, event.getAndonTypeId()));
    }

    @Override
    public List<AndonEvent> findByAndonLevelId(Long andonLevelId) {
        return findWhere(event -> matches(andonLevelId, event.getAndonLevelId()));
    }

    @Override
    public List<AndonEvent> findByWorkshopId(String workshopId) {
        return findWhere(event -> matches(workshopId, event.getWorkshopId()));
    }

    @Override
    public List<AndonEvent> findByEquipmentId(String equipmentId) {
        return findWhere(event -> matches(equipmentId, event.getEquipmentId()));
    }

    @Override
    public List<AndonEvent> findByReporterId(String reporterId) {
        return findWhere(event -> matches(reporterId, event.getReporterId()));
    }

    @Override
    public List<AndonEvent> findByCurrentHandlerId(String currentHandlerId) {
        return findWhere(event -> matches(currentHandlerId, event.getCurrentHandlerId()));
    }

    @Override
    public List<AndonEvent> findByStatusAndOccurredAtBefore(AndonEvent.AndonEventStatus status, LocalDateTime time) {
        if (time == null) {
            return new ArrayList<>();
        }
        return findWhere(event -> event.getStatus() == status
                && event.getOccurredAt() != null
                && event.getOccurredAt().isBefore(time));
    }

    @Override
    public List<AndonEvent> findAll() {
        return new ArrayList<>(store.values());
    }

    @Override
    public AndonEvent save(AndonEvent andonEvent) {
        if (andonEvent.getId() == null) {
            andonEvent.setId(idSeq.getAndIncrement());
        }
        store.put(andonEvent.getId(), andonEvent);
        return andonEvent;
    }

    @Override
    public void update(AndonEvent andonEvent) {
        store.put(andonEvent.getId(), andonEvent);
    }

    @Override
    public void deleteById(Long id) {
        store.remove(id);
    }

    private List<AndonEvent> findWhere(Predicate<AndonEvent> predicate) {
        return store.values().stream().filter(predicate).collect(Collectors.toList());
    }

    private boolean matches(Object expected, Object actual) {
        return expected != null && expected.equals(actual);
    }
}
