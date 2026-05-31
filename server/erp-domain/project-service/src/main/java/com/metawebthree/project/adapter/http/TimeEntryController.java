package com.metawebthree.project.adapter.http;

import com.metawebthree.project.adapter.vo.Result;
import com.metawebthree.project.application.command.timeEntry.TimeEntryCommandService;
import com.metawebthree.project.application.query.timeEntry.TimeEntryQueryService;
import com.metawebthree.project.domain.entity.TimeEntry;
import lombok.RequiredArgsConstructor;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.web.bind.annotation.*;
import java.time.LocalDate;
import java.util.List;

@RestController
@RequestMapping("/project-service/time-entries")
@RequiredArgsConstructor
public class TimeEntryController {

    private final TimeEntryCommandService timeEntryCommandService;
    private final TimeEntryQueryService timeEntryQueryService;

    @PostMapping
    public Result<TimeEntry> create(@RequestBody TimeEntry timeEntry) {
        TimeEntry created = timeEntryCommandService.create(timeEntry);
        return Result.success(created);
    }

    @PutMapping
    public Result<TimeEntry> update(@RequestBody TimeEntry timeEntry) {
        TimeEntry updated = timeEntryCommandService.update(timeEntry);
        return Result.success(updated);
    }

    @DeleteMapping("/{id}")
    public Result<Void> delete(@PathVariable Long id) {
        timeEntryCommandService.delete(id);
        return Result.success(null);
    }

    @GetMapping("/{id}")
    public Result<TimeEntry> getById(@PathVariable Long id) {
        TimeEntry timeEntry = timeEntryQueryService.findById(id);
        return Result.success(timeEntry);
    }

    @GetMapping("/project/{projectId}")
    public Result<List<TimeEntry>> getByProjectId(@PathVariable Long projectId) {
        List<TimeEntry> entries = timeEntryQueryService.findByProjectId(projectId);
        return Result.success(entries);
    }

    @GetMapping("/employee/{employeeId}")
    public Result<List<TimeEntry>> getByEmployeeId(@PathVariable Long employeeId) {
        List<TimeEntry> entries = timeEntryQueryService.findByEmployeeId(employeeId);
        return Result.success(entries);
    }

    @GetMapping("/page")
    public Result<Result.PageResult<TimeEntry>> getPage(
            @RequestParam(defaultValue = "1") int pageNum,
            @RequestParam(defaultValue = "10") int pageSize,
            @RequestParam(required = false) Long projectId,
            @RequestParam(required = false) Long employeeId,
            @RequestParam(required = false) String status,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate startDate,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate endDate) {
        var page = timeEntryQueryService.findPage(pageNum, pageSize, projectId, employeeId, status, startDate, endDate);
        return Result.successPage(page.getRecords(), page.getTotal());
    }

    @PutMapping("/{id}/approve")
    public Result<TimeEntry> approve(@PathVariable Long id, @RequestParam Long approverId, @RequestParam String approverName) {
        TimeEntry approved = timeEntryCommandService.approve(id, approverId, approverName);
        return Result.success(approved);
    }

    @PutMapping("/{id}/reject")
    public Result<TimeEntry> reject(@PathVariable Long id) {
        TimeEntry rejected = timeEntryCommandService.reject(id);
        return Result.success(rejected);
    }
}