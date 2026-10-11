package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.AndonLevel;
import java.util.List;
import java.util.Optional;

public interface AndonLevelRepository {
    Optional<AndonLevel> findById(Long id);
    Optional<AndonLevel> findByLevelCode(String levelCode);
    Optional<AndonLevel> findByLevelValue(Integer levelValue);
    List<AndonLevel> findByStatus(AndonLevel.AndonLevelStatus status);
    List<AndonLevel> findAll();
    AndonLevel save(AndonLevel andonLevel);
    void update(AndonLevel andonLevel);
    void deleteById(Long id);
}