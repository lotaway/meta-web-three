package com.metawebthree.digitaltwin.application.command;

import com.metawebthree.digitaltwin.domain.entity.Shelf;
import com.metawebthree.digitaltwin.domain.entity.Shelf.ShelfStatus;
import com.metawebthree.digitaltwin.domain.repository.ShelfRepository;
import com.metawebthree.digitaltwin.domain.repository.WarehouseRepository;
import com.metawebthree.digitaltwin.infrastructure.event.DigitalTwinEventPublisher;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.time.LocalDateTime;

@Service
public class ShelfCommandService {
    private static final Integer DEFAULT_LEVEL_NUMBER = 1;
    private static final Integer DEFAULT_TOTAL_LEVELS = 3;

    private final ShelfRepository shelfRepository;
    private final WarehouseRepository warehouseRepository;
    private final DigitalTwinEventPublisher eventPublisher;

    public ShelfCommandService(ShelfRepository shelfRepository,
                                WarehouseRepository warehouseRepository,
                                DigitalTwinEventPublisher eventPublisher) {
        this.shelfRepository = shelfRepository;
        this.warehouseRepository = warehouseRepository;
        this.eventPublisher = eventPublisher;
    }

    @Transactional
    public Shelf createShelf(CreateShelfRequest request) {
        validateShelfRequest(request);
        Shelf shelf = new Shelf(request.shelfCode, request.warehouseCode, 
                                request.rowNumber, request.columnNumber);
        initializeShelfFields(shelf, request);
        shelfRepository.insert(shelf);
        return shelf;
    }

    private void validateShelfRequest(CreateShelfRequest request) {
        if (shelfRepository.existsByShelfCode(request.shelfCode)) {
            throw new IllegalArgumentException("Shelf code already exists: " + request.shelfCode);
        }
        if (!warehouseRepository.existsByWarehouseCode(request.warehouseCode)) {
            throw new IllegalArgumentException("Warehouse not found: " + request.warehouseCode);
        }
    }

    private void initializeShelfFields(Shelf shelf, CreateShelfRequest request) {
        shelf.setZone(request.zone);
        shelf.setLevelNumber(DEFAULT_LEVEL_NUMBER);
        shelf.setTotalLevels(request.totalLevels != null ? request.totalLevels : DEFAULT_TOTAL_LEVELS);
        shelf.setStatus(ShelfStatus.EMPTY);
        shelf.setMaxWeight(request.maxWeight);
        shelf.setCurrentWeight(BigDecimal.ZERO);
        shelf.setPositionX(request.positionX);
        shelf.setPositionY(request.positionY);
        shelf.setPositionZ(request.positionZ);
        shelf.setRotationY(request.rotationY);
        shelf.setLength(request.length);
        shelf.setWidth(request.width);
        shelf.setHeight(request.height);
        shelf.setCreatedAt(LocalDateTime.now());
        shelf.setUpdatedAt(LocalDateTime.now());
    }

    @Transactional
    public Shelf updateShelf(UpdateShelfRequest request) {
        Shelf shelf = shelfRepository.findById(request.id)
                .orElseThrow(() -> new IllegalArgumentException("Shelf not found: " + request.id));
        if (request.zone != null) {
            shelf.setZone(request.zone);
        }
        if (request.status != null) {
            shelf.setStatus(request.status);
        }
        if (request.maxWeight != null) {
            shelf.setMaxWeight(request.maxWeight);
        }
        shelf.setUpdatedAt(LocalDateTime.now());
        shelfRepository.update(shelf);
        return shelf;
    }

    @Transactional
    public void occupyShelf(Long id) {
        Shelf shelf = shelfRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Shelf not found: " + id));
        shelf.occupy();
        shelfRepository.update(shelf);
        eventPublisher.publishShelfStatusChanged(
                shelf.getWarehouseCode(),
                shelf.getShelfCode(),
                shelf.getStatus().name());
    }

    @Transactional
    public void clearShelf(Long id) {
        Shelf shelf = shelfRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Shelf not found: " + id));
        shelf.clear();
        shelfRepository.update(shelf);
        eventPublisher.publishShelfStatusChanged(
                shelf.getWarehouseCode(),
                shelf.getShelfCode(),
                shelf.getStatus().name());
    }

    @Transactional
    public void deleteShelf(Long id) {
        Shelf shelf = shelfRepository.findById(id)
                .orElseThrow(() -> new IllegalArgumentException("Shelf not found: " + id));
        shelfRepository.delete(shelf);
    }

    public static class CreateShelfRequest {
        public String shelfCode;
        public String warehouseCode;
        public String zone;
        public Integer rowNumber;
        public Integer columnNumber;
        public Integer totalLevels;
        public BigDecimal maxWeight;
        public BigDecimal positionX;
        public BigDecimal positionY;
        public BigDecimal positionZ;
        public BigDecimal rotationY;
        public BigDecimal length;
        public BigDecimal width;
        public BigDecimal height;
    }

    public static class UpdateShelfRequest {
        public Long id;
        public String zone;
        public ShelfStatus status;
        public BigDecimal maxWeight;
    }
}