package com.metawebthree.mes.infrastructure.persistence.repository;

import com.metawebthree.mes.domain.entity.WorkOrder;
import com.metawebthree.mes.domain.repository.WorkOrderRepository;
import org.springframework.stereotype.Repository;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

@Repository
public class WorkOrderRepositoryImpl implements WorkOrderRepository {
    private final Map<Long, WorkOrder> storage = new ConcurrentHashMap<>();
    private final AtomicLong idGen = new AtomicLong(1);

    @Override
    public Optional<WorkOrder> findById(Long id) { return Optional.ofNullable(storage.get(id)); }
    @Override
    public Optional<WorkOrder> findByWorkOrderNo(String no) {
        return storage.values().stream().filter(w -> w.getWorkOrderNo().equals(no)).findFirst();
    }
    @Override
    public List<WorkOrder> findByStatus(WorkOrder.WorkOrderStatus status) {
        return storage.values().stream().filter(w -> w.getStatus() == status).collect(Collectors.toList());
    }
    @Override
    public List<WorkOrder> findByWorkshopId(String workshopId) {
        return storage.values().stream().filter(w -> w.getWorkshopId().equals(workshopId)).collect(Collectors.toList());
    }
    @Override
    public List<WorkOrder> findByProductCode(String productCode) {
        return storage.values().stream().filter(w -> w.getProductCode().equals(productCode)).collect(Collectors.toList());
    }
    @Override
    public WorkOrder save(WorkOrder w) { if (w.getId() == null) w.setId(idGen.getAndIncrement()); storage.put(w.getId(), w); return w; }
    @Override
    public void update(WorkOrder w) { if (w.getId() != null && storage.containsKey(w.getId())) storage.put(w.getId(), w); }
    @Override
    public void deleteById(Long id) { storage.remove(id); }
}