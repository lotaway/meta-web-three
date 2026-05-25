package com.metawebthree.digitaltwin.application.query;

import com.metawebthree.digitaltwin.domain.entity.Warehouse;
import com.metawebthree.digitaltwin.domain.entity.Warehouse.WarehouseStatus;
import com.metawebthree.digitaltwin.domain.repository.WarehouseRepository;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
public class WarehouseQueryService {

    private final WarehouseRepository warehouseRepository;

    public WarehouseQueryService(WarehouseRepository warehouseRepository) {
        this.warehouseRepository = warehouseRepository;
    }

    public Optional<Warehouse> findById(Long id) {
        return warehouseRepository.findById(id);
    }

    public Optional<Warehouse> findByWarehouseCode(String warehouseCode) {
        return warehouseRepository.findByWarehouseCode(warehouseCode);
    }

    public List<Warehouse> findAll() {
        return warehouseRepository.findAll();
    }

    public List<Warehouse> findByStatus(WarehouseStatus status) {
        return warehouseRepository.findByStatus(status);
    }

    public BigDecimal calculateUtilizationRate(String warehouseCode) {
        return warehouseRepository.findByWarehouseCode(warehouseCode)
                .map(Warehouse::calculateUtilizationRate)
                .orElse(BigDecimal.ZERO);
    }

    public static class WarehouseSummary {
        public Long id;
        public String warehouseCode;
        public String warehouseName;
        public String status;
        public BigDecimal totalArea;
        public BigDecimal usedArea;
        public BigDecimal utilizationRate;
        public String location;
    }

    public List<WarehouseSummary> getWarehouseSummaries() {
        return warehouseRepository.findAll().stream()
                .map(wh -> {
                    WarehouseSummary summary = new WarehouseSummary();
                    summary.id = wh.getId();
                    summary.warehouseCode = wh.getWarehouseCode();
                    summary.warehouseName = wh.getWarehouseName();
                    summary.status = wh.getStatus() != null ? wh.getStatus().name() : null;
                    summary.totalArea = wh.getTotalArea();
                    summary.usedArea = wh.getUsedArea();
                    summary.utilizationRate = wh.calculateUtilizationRate();
                    summary.location = wh.getLocation();
                    return summary;
                })
                .collect(Collectors.toList());
    }
}