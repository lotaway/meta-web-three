package com.metawebthree.project.infrastructure.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.project.domain.entity.TimeEntry;
import com.metawebthree.project.domain.repository.timeEntry.TimeEntryRepository;
import com.metawebthree.project.infrastructure.mapper.TimeEntryMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Repository;
import java.time.LocalDate;
import java.util.List;

@Repository
@RequiredArgsConstructor
public class TimeEntryRepositoryImpl implements TimeEntryRepository {

    private final TimeEntryMapper timeEntryMapper;

    @Override
    public TimeEntry save(TimeEntry timeEntry) {
        timeEntryMapper.insert(timeEntry);
        return timeEntry;
    }

    @Override
    public TimeEntry update(TimeEntry timeEntry) {
        timeEntryMapper.updateById(timeEntry);
        return timeEntry;
    }

    @Override
    public void delete(Long id) {
        timeEntryMapper.deleteById(id);
    }

    @Override
    public TimeEntry findById(Long id) {
        return timeEntryMapper.selectById(id);
    }

    @Override
    public List<TimeEntry> findByProjectId(Long projectId) {
        LambdaQueryWrapper<TimeEntry> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(TimeEntry::getProjectId, projectId).orderByDesc(TimeEntry::getWorkDate);
        return timeEntryMapper.selectList(wrapper);
    }

    @Override
    public List<TimeEntry> findByEmployeeId(Long employeeId) {
        LambdaQueryWrapper<TimeEntry> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(TimeEntry::getEmployeeId, employeeId).orderByDesc(TimeEntry::getWorkDate);
        return timeEntryMapper.selectList(wrapper);
    }

    @Override
    public List<TimeEntry> findByDateRange(LocalDate startDate, LocalDate endDate) {
        LambdaQueryWrapper<TimeEntry> wrapper = new LambdaQueryWrapper<>();
        wrapper.between(TimeEntry::getWorkDate, startDate, endDate)
               .orderByDesc(TimeEntry::getWorkDate);
        return timeEntryMapper.selectList(wrapper);
    }

    @Override
    public IPage<TimeEntry> findPage(Page<TimeEntry> page, Long projectId, Long employeeId, String status, LocalDate startDate, LocalDate endDate) {
        LambdaQueryWrapper<TimeEntry> wrapper = new LambdaQueryWrapper<>();
        if (projectId != null) {
            wrapper.eq(TimeEntry::getProjectId, projectId);
        }
        if (employeeId != null) {
            wrapper.eq(TimeEntry::getEmployeeId, employeeId);
        }
        if (status != null && !status.isEmpty()) {
            wrapper.eq(TimeEntry::getStatus, status);
        }
        if (startDate != null && endDate != null) {
            wrapper.between(TimeEntry::getWorkDate, startDate, endDate);
        }
        wrapper.orderByDesc(TimeEntry::getWorkDate);
        return timeEntryMapper.selectPage(page, wrapper);
    }
}