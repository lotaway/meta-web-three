package com.metawebthree.project.domain.repository.timeEntry;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.project.domain.entity.TimeEntry;
import java.time.LocalDate;
import java.util.List;

public interface TimeEntryRepository {
    TimeEntry save(TimeEntry timeEntry);
    TimeEntry update(TimeEntry timeEntry);
    void delete(Long id);
    TimeEntry findById(Long id);
    List<TimeEntry> findByProjectId(Long projectId);
    List<TimeEntry> findByEmployeeId(Long employeeId);
    List<TimeEntry> findByDateRange(LocalDate startDate, LocalDate endDate);
    IPage<TimeEntry> findPage(Page<TimeEntry> page, Long projectId, Long employeeId, String status, LocalDate startDate, LocalDate endDate);
}