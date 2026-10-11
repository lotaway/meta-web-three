package com.metawebthree.digitaltwin.application.query;

import com.metawebthree.digitaltwin.domain.entity.Shelf;
import com.metawebthree.digitaltwin.domain.entity.Shelf.ShelfStatus;
import com.metawebthree.digitaltwin.domain.repository.ShelfRepository;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
public class ShelfQueryService {

    private final ShelfRepository shelfRepository;

    public ShelfQueryService(ShelfRepository shelfRepository) {
        this.shelfRepository = shelfRepository;
    }

    public Optional<Shelf> findById(Long id) {
        return shelfRepository.findById(id);
    }

    public Optional<Shelf> findByShelfCode(String shelfCode) {
        return shelfRepository.findByShelfCode(shelfCode);
    }

    public List<Shelf> findAll() {
        return shelfRepository.findAll();
    }

    public List<Shelf> findByWarehouseCode(String warehouseCode) {
        return shelfRepository.findByWarehouseCode(warehouseCode);
    }

    public List<Shelf> findByWarehouseCodeAndStatus(String warehouseCode, ShelfStatus status) {
        return shelfRepository.findByWarehouseCodeAndStatus(warehouseCode, status);
    }

    public List<Shelf> findByWarehouseCodeAndZone(String warehouseCode, String zone) {
        return shelfRepository.findByWarehouseCodeAndZone(warehouseCode, zone);
    }

    public List<Shelf> findEmptyShelves(String warehouseCode) {
        return shelfRepository.findEmptyShelves(warehouseCode);
    }

    public static class ShelfSummary {
        public Long id;
        public String shelfCode;
        public String warehouseCode;
        public String zone;
        public Integer rowNumber;
        public Integer columnNumber;
        public String status;
        public BigDecimal maxWeight;
        public BigDecimal currentWeight;
    }

    public List<ShelfSummary> getShelfSummaries(String warehouseCode) {
        return shelfRepository.findByWarehouseCode(warehouseCode).stream()
                .map(shelf -> {
                    ShelfSummary summary = new ShelfSummary();
                    summary.id = shelf.getId();
                    summary.shelfCode = shelf.getShelfCode();
                    summary.warehouseCode = shelf.getWarehouseCode();
                    summary.zone = shelf.getZone();
                    summary.rowNumber = shelf.getRowNumber();
                    summary.columnNumber = shelf.getColumnNumber();
                    summary.status = shelf.getStatus() != null ? shelf.getStatus().name() : null;
                    summary.maxWeight = shelf.getMaxWeight();
                    summary.currentWeight = shelf.getCurrentWeight();
                    return summary;
                })
                .collect(Collectors.toList());
    }
}