package com.metawebthree.digitaltwin.infrastructure.persistence.repository;

import com.metawebthree.digitaltwin.domain.entity.Workshop;
import com.metawebthree.digitaltwin.domain.repository.WorkshopRepository;
import org.springframework.stereotype.Repository;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

@Repository
public class WorkshopRepositoryImpl implements WorkshopRepository {
    private final Map<Long, Workshop> storage = new ConcurrentHashMap<>();
    private final AtomicLong idGen = new AtomicLong(1);

    @Override
    public Optional<Workshop> findById(Long id) { return Optional.ofNullable(storage.get(id)); }

    @Override
    public Optional<Workshop> findByWorkshopCode(String code) {
        return storage.values().stream().filter(w -> w.getWorkshopCode().equals(code)).findFirst();
    }

    @Override
    public List<Workshop> findByStatus(Workshop.WorkshopStatus status) {
        return storage.values().stream().filter(w -> w.getStatus() == status).collect(Collectors.toList());
    }

    @Override
    public List<Workshop> findAll() { return new ArrayList<>(storage.values()); }

    @Override
    public Workshop save(Workshop w) { if (w.getId() == null) w.setId(idGen.getAndIncrement()); storage.put(w.getId(), w); return w; }

    @Override
    public void update(Workshop w) { if (w.getId() != null && storage.containsKey(w.getId())) storage.put(w.getId(), w); }

    @Override
    public void deleteById(Long id) { storage.remove(id); }
}