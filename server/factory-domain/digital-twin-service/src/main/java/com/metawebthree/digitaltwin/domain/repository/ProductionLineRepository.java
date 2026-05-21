package com.metawebthree.digitaltwin.domain.repository;

import com.metawebthree.digitaltwin.domain.entity.ProductionLine;
import java.util.List;
import java.util.Optional;

public interface ProductionLineRepository {
    Optional<ProductionLine> findById(Long id);
    Optional<ProductionLine> findByLineCode(String lineCode);
    List<ProductionLine> findByWorkshopId(String workshopId);
    List<ProductionLine> findByStatus(ProductionLine.ProductionLineStatus status);
    List<ProductionLine> findAll();
    ProductionLine save(ProductionLine line);
    void update(ProductionLine line);
    void deleteById(Long id);
}