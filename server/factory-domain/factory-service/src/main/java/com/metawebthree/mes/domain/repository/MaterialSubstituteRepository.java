package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.MaterialSubstitute;
import java.util.List;
import java.util.Optional;

public interface MaterialSubstituteRepository {
    Optional<MaterialSubstitute> findById(Long id);
    Optional<MaterialSubstitute> findByProductCodeAndMainMaterialCode(String productCode, String mainMaterialCode);
    List<MaterialSubstitute> findByProductCode(String productCode);
    List<MaterialSubstitute> findActiveByProductCode(String productCode);
    MaterialSubstitute save(MaterialSubstitute substitute);
    void update(MaterialSubstitute substitute);
    void deleteById(Long id);
}