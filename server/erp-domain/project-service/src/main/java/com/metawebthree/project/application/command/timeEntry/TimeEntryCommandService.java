package com.metawebthree.project.application.command.timeEntry;

import com.metawebthree.project.domain.entity.TimeEntry;
import com.metawebthree.project.domain.exception.TimeEntryNotFoundException;
import com.metawebthree.project.domain.repository.timeEntry.TimeEntryRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import java.time.LocalDateTime;

@Service
@RequiredArgsConstructor
public class TimeEntryCommandService {

    private final TimeEntryRepository timeEntryRepository;

    @Transactional
    public TimeEntry create(TimeEntry timeEntry) {
        timeEntry.setStatus("PENDING");
        timeEntry.setCreatedAt(LocalDateTime.now());
        timeEntry.setUpdatedAt(LocalDateTime.now());
        return timeEntryRepository.save(timeEntry);
    }

    @Transactional
    public TimeEntry update(TimeEntry timeEntry) {
        TimeEntry existing = timeEntryRepository.findById(timeEntry.getId());
        if (existing == null) {
            throw new TimeEntryNotFoundException("TimeEntry not found: " + timeEntry.getId());
        }
        timeEntry.setCreatedAt(existing.getCreatedAt());
        timeEntry.setCreatedBy(existing.getCreatedBy());
        timeEntry.setUpdatedAt(LocalDateTime.now());
        return timeEntryRepository.update(timeEntry);
    }

    @Transactional
    public void delete(Long id) {
        timeEntryRepository.delete(id);
    }

    @Transactional
    public TimeEntry approve(Long id, Long approverId, String approverName) {
        TimeEntry timeEntry = timeEntryRepository.findById(id);
        if (timeEntry == null) {
            throw new TimeEntryNotFoundException("TimeEntry not found: " + id);
        }
        timeEntry.setStatus("APPROVED");
        timeEntry.setApproverId(approverId);
        timeEntry.setApproverName(approverName);
        timeEntry.setApprovedAt(LocalDateTime.now());
        timeEntry.setUpdatedAt(LocalDateTime.now());
        return timeEntryRepository.update(timeEntry);
    }

    @Transactional
    public TimeEntry reject(Long id) {
        TimeEntry timeEntry = timeEntryRepository.findById(id);
        if (timeEntry == null) {
            throw new TimeEntryNotFoundException("TimeEntry not found: " + id);
        }
        timeEntry.setStatus("REJECTED");
        timeEntry.setUpdatedAt(LocalDateTime.now());
        return timeEntryRepository.update(timeEntry);
    }
}