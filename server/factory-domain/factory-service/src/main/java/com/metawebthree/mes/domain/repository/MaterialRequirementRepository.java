package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.MaterialRequirement;
import java.util.List;
import java.util.Optional;

public interface MaterialRequirementRepository {
    Optional<MaterialRequirement> findById(Long id);
    Optional<MaterialRequirement> findByRequirementNo(String requirementNo);
    Optional<MaterialRequirement> findByWorkOrderNo(String workOrderNo);
    List<MaterialRequirement> findByStatus(String status);
    List<MaterialRequirement> findByWarehouseId(String warehouseId);
    List<MaterialRequirement> findByWorkshopId(String workshopId);
    List<MaterialRequirement> findByProductCode(String productCode);
    MaterialRequirement save(MaterialRequirement requirement);
    void update(MaterialRequirement requirement);
    void deleteById(Long id);
}