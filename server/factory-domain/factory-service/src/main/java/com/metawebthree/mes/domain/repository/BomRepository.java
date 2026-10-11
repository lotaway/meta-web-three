package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.BomBillOfMaterials;
import com.metawebthree.mes.domain.entity.BomVersion;
import java.util.List;
import java.util.Optional;

public interface BomRepository {
    Optional<BomBillOfMaterials> findById(Long id);
    Optional<BomBillOfMaterials> findByBomCode(String bomCode);
    List<BomBillOfMaterials> findByProductCode(String productCode);
    List<BomBillOfMaterials> findActiveByProductCode(String productCode);
    List<BomBillOfMaterials> findByProductCodeAndVersion(String productCode, String version);
    BomBillOfMaterials save(BomBillOfMaterials bom);
    void update(BomBillOfMaterials bom);
    void deleteById(Long id);
    
    // BOM版本管理
    Optional<BomVersion> findVersionByProductCode(String productCode);
    BomVersion saveVersion(BomVersion version);
    
    // 工序BOM
    List<com.metawebthree.mes.domain.entity.ProcessBomItem> findProcessBomByRouteId(String processRouteId);
    List<com.metawebthree.mes.domain.entity.ProcessBomItem> findProcessBomByProcessCode(String processRouteId, String processCode);
}