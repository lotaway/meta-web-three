package com.metawebthree.finance.domain.repository;

import com.metawebthree.finance.domain.entity.FinancialRatio;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

public interface FinancialRatioRepository {
    Optional<FinancialRatio> findById(Long id);
    List<FinancialRatio> findByRatioType(String ratioType);
    List<FinancialRatio> findByPeriod(String period);
    List<FinancialRatio> findByCalculatedAtBetween(LocalDateTime start, LocalDateTime end);
    List<FinancialRatio> findAll();
    void save(FinancialRatio ratio);
    void update(FinancialRatio ratio);
    void delete(Long id);
}