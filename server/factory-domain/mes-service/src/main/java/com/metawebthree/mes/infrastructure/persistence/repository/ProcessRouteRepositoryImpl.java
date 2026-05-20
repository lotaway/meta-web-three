package com.metawebthree.mes.infrastructure.persistence.repository;

import com.metawebthree.mes.domain.entity.ProcessRoute;
import com.metawebthree.mes.domain.repository.ProcessRouteRepository;
import org.springframework.stereotype.Repository;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

@Repository
public class ProcessRouteRepositoryImpl implements ProcessRouteRepository {
    private final Map<Long, ProcessRoute> storage = new ConcurrentHashMap<>();
    private final AtomicLong idGen = new AtomicLong(1);

    @Override
    public Optional<ProcessRoute> findById(Long id) { return Optional.ofNullable(storage.get(id)); }
    @Override
    public Optional<ProcessRoute> findByRouteCode(String code) {
        return storage.values().stream().filter(r -> r.getRouteCode().equals(code)).findFirst();
    }
    @Override
    public List<ProcessRoute> findByProductCode(String productCode) {
        return storage.values().stream().filter(r -> r.getProductCode().equals(productCode)).collect(Collectors.toList());
    }
    @Override
    public List<ProcessRoute> findByStatus(ProcessRoute.RouteStatus status) {
        return storage.values().stream().filter(r -> r.getStatus() == status).collect(Collectors.toList());
    }
    @Override
    public ProcessRoute save(ProcessRoute r) { if (r.getId() == null) r.setId(idGen.getAndIncrement()); storage.put(r.getId(), r); return r; }
    @Override
    public void update(ProcessRoute r) { if (r.getId() != null && storage.containsKey(r.getId())) storage.put(r.getId(), r); }
    @Override
    public void deleteById(Long id) { storage.remove(id); }
}