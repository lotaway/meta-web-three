package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.SpcControlChart;
import java.util.List;
import java.util.Optional;

public interface SpcControlChartRepository {
    Optional<SpcControlChart> findById(Long id);
    Optional<SpcControlChart> findByChartCode(String chartCode);
    List<SpcControlChart> findAll();
    List<SpcControlChart> findByChartType(SpcControlChart.ChartType chartType);
    List<SpcControlChart> findByIsEnabled(Boolean isEnabled);
    SpcControlChart save(SpcControlChart chart);
    void update(SpcControlChart chart);
    void deleteById(Long id);
    Boolean existsByChartCode(String chartCode);
}