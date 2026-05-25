package com.metawebthree.digitaltwin.domain.repository;

import com.metawebthree.digitaltwin.domain.entity.Shelf;
import com.metawebthree.digitaltwin.domain.entity.Shelf.ShelfStatus;

import java.util.List;
import java.util.Optional;

public interface ShelfRepository {
    Optional<Shelf> findById(Long id);
    Optional<Shelf> findByShelfCode(String shelfCode);
    List<Shelf> findAll();
    List<Shelf> findByWarehouseCode(String warehouseCode);
    List<Shelf> findByWarehouseCodeAndStatus(String warehouseCode, ShelfStatus status);
    List<Shelf> findByWarehouseCodeAndZone(String warehouseCode, String zone);
    List<Shelf> findEmptyShelves(String warehouseCode);
    Shelf save(Shelf shelf);
    void delete(Shelf shelf);
    boolean existsByShelfCode(String shelfCode);
}