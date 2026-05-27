package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.AndonType;
import java.util.List;
import java.util.Optional;

public interface AndonTypeRepository {
    Optional<AndonType> findById(Long id);
    Optional<AndonType> findByTypeCode(String typeCode);
    List<AndonType> findByCategory(AndonType.AndonCategory category);
    List<AndonType> findByStatus(AndonType.AndonStatus status);
    List<AndonType> findAll();
    AndonType save(AndonType andonType);
    void update(AndonType andonType);
    void deleteById(Long id);
}