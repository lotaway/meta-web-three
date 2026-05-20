package com.metawebthree.mes.infrastructure.persistence.repository;

import com.metawebthree.mes.domain.entity.ProductionTask;
import com.metawebthree.mes.domain.repository.ProductionTaskRepository;
import org.springframework.stereotype.Repository;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

@Repository
public class ProductionTaskRepositoryImpl implements ProductionTaskRepository {
    private final Map<Long, ProductionTask> storage = new ConcurrentHashMap<>();
    private final AtomicLong idGen = new AtomicLong(1);

    @Override
    public Optional<ProductionTask> findById(Long id) { return Optional.ofNullable(storage.get(id)); }
    @Override
    public Optional<ProductionTask> findByTaskNo(String no) {
        return storage.values().stream().filter(t -> t.getTaskNo().equals(no)).findFirst();
    }
    @Override
    public List<ProductionTask> findByWorkOrderId(Long workOrderId) {
        return storage.values().stream().filter(t -> t.getWorkOrderId().equals(workOrderId)).collect(Collectors.toList());
    }
    @Override
    public List<ProductionTask> findByStatus(ProductionTask.TaskStatus status) {
        return storage.values().stream().filter(t -> t.getStatus() == status).collect(Collectors.toList());
    }
    @Override
    public List<ProductionTask> findByWorkstationId(String workstationId) {
        return storage.values().stream().filter(t -> t.getWorkstationId().equals(workstationId)).collect(Collectors.toList());
    }
    @Override
    public ProductionTask save(ProductionTask t) { if (t.getId() == null) t.setId(idGen.getAndIncrement()); storage.put(t.getId(), t); return t; }
    @Override
    public void update(ProductionTask t) { if (t.getId() != null && storage.containsKey(t.getId())) storage.put(t.getId(), t); }
    @Override
    public void deleteById(Long id) { storage.remove(id); }
}