package com.metawebthree.digitaltwin.infrastructure.persistence.repository;

import com.metawebthree.digitaltwin.domain.entity.Alert;
import com.metawebthree.digitaltwin.domain.repository.AlertRepository;
import org.springframework.stereotype.Repository;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

@Repository
public class AlertRepositoryImpl implements AlertRepository {
    private final Map<Long, Alert> storage = new ConcurrentHashMap<>();
    private final AtomicLong idGen = new AtomicLong(1);

    @Override
    public Optional<Alert> findById(Long id) { return Optional.ofNullable(storage.get(id)); }

    @Override
    public Optional<Alert> findByAlertCode(String code) {
        return storage.values().stream().filter(a -> a.getAlertCode().equals(code)).findFirst();
    }

    @Override
    public List<Alert> findByDeviceCode(String deviceCode) {
        return storage.values().stream().filter(a -> a.getDeviceCode().equals(deviceCode)).collect(Collectors.toList());
    }

    @Override
    public List<Alert> findByWorkshopId(String workshopId) {
        return storage.values().stream().filter(a -> a.getWorkshopId().equals(workshopId)).collect(Collectors.toList());
    }

    @Override
    public List<Alert> findByStatus(Alert.AlertStatus status) {
        return storage.values().stream().filter(a -> a.getStatus() == status).collect(Collectors.toList());
    }

    @Override
    public List<Alert> findByLevel(Alert.AlertLevel level) {
        return storage.values().stream().filter(a -> a.getLevel() == level).collect(Collectors.toList());
    }

    @Override
    public List<Alert> findAll() { return new ArrayList<>(storage.values()); }

    @Override
    public Alert save(Alert a) { if (a.getId() == null) a.setId(idGen.getAndIncrement()); storage.put(a.getId(), a); return a; }

    @Override
    public void update(Alert a) { if (a.getId() != null && storage.containsKey(a.getId())) storage.put(a.getId(), a); }

    @Override
    public void deleteById(Long id) { storage.remove(id); }
}