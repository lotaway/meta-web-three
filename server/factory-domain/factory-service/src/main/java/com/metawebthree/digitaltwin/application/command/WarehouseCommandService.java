package com.metawebthree.digitaltwin.application.command;

import com.metawebthree.digitaltwin.domain.entity.Warehouse;
import com.metawebthree.digitaltwin.domain.entity.Warehouse.WarehouseStatus;
import com.metawebthree.digitaltwin.domain.repository.WarehouseRepository;
import com.metawebthree.digitaltwin.infrastructure.event.DigitalTwinEventPublisher;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.time.LocalDateTime;

@Service
public class WarehouseCommandService {

    private final WarehouseRepository warehouseRepository;
    private final DigitalTwinEventPublisher eventPublisher;

    public WarehouseCommandService(
            WarehouseRepository warehouseRepository,
            DigitalTwinEventPublisher eventPublisher) {
        this.warehouseRepository = warehouseRepository;
        this.eventPublisher = eventPublisher;
    }

    @Transactional
    public Warehouse createWarehouse(CreateWarehouseRequest request) {
        if (warehouseRepository.existsByWarehouseCode(request.warehouseCode)) {
            throw new IllegalArgumentException("Warehouse code already exists: " + request.warehouseCode);
        }
        Warehouse warehouse = new Warehouse(request.warehouseCode, request.warehouseName);
        warehouse.setDescription(request.description);
        warehouse.setStatus(WarehouseStatus.PLANNING);
        warehouse.setTotalArea(request.totalArea);
        warehouse.setUsedArea(BigDecimal.ZERO);
        warehouse.setLocation(request.location);
        warehouse.setCenterX(request.centerX);
        warehouse.setCenterY(request.centerY);
        warehouse.setCenterZ(request.centerZ);
        warehouse.setWidth(request.width);
        warehouse.setLength(request.length);
        warehouse.setHeight(request.height);
        warehouse.setCreatedAt(LocalDateTime.now());
        warehouse.setUpdatedAt(LocalDateTime.now());
        warehouseRepository.insert(warehouse);
        return warehouse;
    }

    @Transactional
    public Warehouse updateWarehouse(UpdateWarehouseRequest request) {
        Warehouse warehouse = warehouseRepository.findById(request.id)
                .orElseThrow(() -> new IllegalArgumentException("Warehouse not found: " + request.id));
        applyUpdates(warehouse, request);
        warehouseRepository.update(warehouse);
        return warehouse;
    }

    private void applyUpdates(Warehouse warehouse, UpdateWarehouseRequest request) {
        if (request.warehouseName != null) {
            warehouse.setWarehouseName(request.warehouseName);
        }
        if (request.description != null) {
            warehouse.setDescription(request.description);
        }
        if (request.status != null) {
            warehouse.setStatus(request.status);
        }
        if (request.totalArea != null) {
            warehouse.setTotalArea(request.totalArea);
        }
        if (request.usedArea != null) {
            warehouse.setUsedArea(request.usedArea);
        }
        if (request.location != null) {
            warehouse.setLocation(request.location);
        }
        warehouse.setUpdatedAt(LocalDateTime.now());
    }

    @Transactional
    public void activateWarehouse(Long id) {
        Warehouse warehouse = warehouseRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Warehouse not found: " + id));
        warehouse.activate();
        warehouseRepository.update(warehouse);
        eventPublisher.publishWarehouseStatusChanged(
                warehouse.getWarehouseCode(),
                warehouse.getStatus().name());
    }

    @Transactional
    public void decommissionWarehouse(Long id) {
        Warehouse warehouse = warehouseRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Warehouse not found: " + id));
        warehouse.decommission();
        warehouseRepository.update(warehouse);
        eventPublisher.publishWarehouseStatusChanged(
                warehouse.getWarehouseCode(),
                warehouse.getStatus().name());
    }

    @Transactional
    public void deleteWarehouse(Long id) {
        Warehouse warehouse = warehouseRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Warehouse not found: " + id));
        warehouseRepository.delete(warehouse);
    }

    public static class CreateWarehouseRequest {
        public String warehouseCode;
        public String warehouseName;
        public String description;
        public BigDecimal totalArea;
        public String location;
        public BigDecimal centerX;
        public BigDecimal centerY;
        public BigDecimal centerZ;
        public BigDecimal width;
        public BigDecimal length;
        public BigDecimal height;
    }

    public static class UpdateWarehouseRequest {
        public Long id;
        public String warehouseName;
        public String description;
        public WarehouseStatus status;
        public BigDecimal totalArea;
        public BigDecimal usedArea;
        public String location;
    }
}