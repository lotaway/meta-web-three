package com.metawebthree.project.application.query.timeEntry;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.project.domain.entity.TimeEntry;
import com.metawebthree.project.domain.repository.timeEntry.TimeEntryRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import java.time.LocalDate;
import java.util.List;

@Service
@RequiredArgsConstructor
public class TimeEntryQueryService {

    private final TimeEntryRepository timeEntryRepository;

    public TimeEntry findById(Long id) {
        return timeEntryRepository.findById(id);
    }

    public List<TimeEntry> findByProjectId(Long projectId) {
        return timeEntryRepository.findByProjectId(projectId);
    }

    public List<TimeEntry> findByEmployeeId(Long employeeId) {
        return timeEntryRepository.findByEmployeeId(employeeId);
    }

    public List<TimeEntry> findByDateRange(LocalDate startDate, LocalDate endDate) {
        return timeEntryRepository.findByDateRange(startDate, endDate);
    }

    public IPage<TimeEntry> findPage(int pageNum, int pageSize, Long projectId, Long employeeId, String status, LocalDate startDate, LocalDate endDate) {
        Page<TimeEntry> page = new Page<>(pageNum, pageSize);
        return timeEntryRepository.findPage(page, projectId, employeeId, status, startDate, endDate);
    }
}