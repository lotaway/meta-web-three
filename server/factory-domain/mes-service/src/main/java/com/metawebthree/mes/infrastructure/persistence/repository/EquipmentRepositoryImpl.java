package com.metawebthree.mes.infrastructure.persistence.repository;

import com.metawebthree.mes.domain.entity.Equipment;
import com.metawebthree.mes.domain.repository.EquipmentRepository;
import org.springframework.stereotype.Repository;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

@Repository
public class EquipmentRepositoryImpl implements EquipmentRepository {
    private final Map<Long, Equipment> storage = new ConcurrentHashMap<>();
    private final AtomicLong idGen = new AtomicLong(1);

    @Override
    public Optional<Equipment> findById(Long id) { return Optional.ofNullable(storage.get(id)); }
    @Override
    public Optional<Equipment> findByEquipmentCode(String code) {
        return storage.values().stream().filter(e -> e.getEquipmentCode().equals(code)).findFirst();
    }
    @Override
    public List<Equipment> findByWorkshopId(String workshopId) {
        return storage.values().stream().filter(e -> e.getWorkshopId().equals(workshopId)).collect(Collectors.toList());
    }
    @Override
    public List<Equipment> findByStatus(Equipment.EquipmentStatus status) {
        return storage.values().stream().filter(e -> e.getStatus() == status).collect(Collectors.toList());
    }
    @Override
    public List<Equipment> findByWorkstationId(String workstationId) {
        return storage.values().stream().filter(e -> e.getWorkstationId().equals(workstationId)).collect(Collectors.toList());
    }
    @Override
    public Equipment save(Equipment e) { if (e.getId() == null) e.setId(idGen.getAndIncrement()); storage.put(e.getId(), e); return e; }
    @Override
    public void update(Equipment e) { if (e.getId() != null && storage.containsKey(e.getId())) storage.put(e.getId(), e); }
    @Override
    public void deleteById(Long id) { storage.remove(id); }
}