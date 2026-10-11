package com.metawebthree.project.domain.repository.costRecord;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.project.domain.entity.CostRecord;
import java.time.LocalDate;
import java.util.List;

public interface CostRecordRepository {
    CostRecord save(CostRecord costRecord);
    CostRecord update(CostRecord costRecord);
    void delete(Long id);
    CostRecord findById(Long id);
    List<CostRecord> findByProjectId(Long projectId);
    List<CostRecord> findByDateRange(LocalDate startDate, LocalDate endDate);
    IPage<CostRecord> findPage(Page<CostRecord> page, Long projectId, String costType, String status, LocalDate startDate, LocalDate endDate);
}