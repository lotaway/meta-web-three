package com.metawebthree.digitaltwin.infrastructure.persistence.repository;

import com.metawebthree.digitaltwin.domain.entity.ProductionLine;
import com.metawebthree.digitaltwin.domain.repository.ProductionLineRepository;
import org.springframework.stereotype.Repository;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

@Repository
public class ProductionLineRepositoryImpl implements ProductionLineRepository {
    private final Map<Long, ProductionLine> storage = new ConcurrentHashMap<>();
    private final AtomicLong idGen = new AtomicLong(1);

    @Override
    public Optional<ProductionLine> findById(Long id) { return Optional.ofNullable(storage.get(id)); }

    @Override
    public Optional<ProductionLine> findByLineCode(String code) {
        return storage.values().stream().filter(l -> l.getLineCode().equals(code)).findFirst();
    }

    @Override
    public List<ProductionLine> findByWorkshopId(String workshopId) {
        return storage.values().stream().filter(l -> l.getWorkshopId().equals(workshopId)).collect(Collectors.toList());
    }

    @Override
    public List<ProductionLine> findByStatus(ProductionLine.ProductionLineStatus status) {
        return storage.values().stream().filter(l -> l.getStatus() == status).collect(Collectors.toList());
    }

    @Override
    public List<ProductionLine> findAll() { return new ArrayList<>(storage.values()); }

    @Override
    public ProductionLine save(ProductionLine l) { if (l.getId() == null) l.setId(idGen.getAndIncrement()); storage.put(l.getId(), l); return l; }

    @Override
    public void update(ProductionLine l) { if (l.getId() != null && storage.containsKey(l.getId())) storage.put(l.getId(), l); }

    @Override
    public void deleteById(Long id) { storage.remove(id); }
}