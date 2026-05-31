package com.metawebthree.project.adapter.http;

import com.metawebthree.project.adapter.vo.Result;
import com.metawebthree.project.application.command.task.TaskCommandService;
import com.metawebthree.project.application.query.task.TaskQueryService;
import com.metawebthree.project.domain.entity.Task;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@RestController
@RequestMapping("/project-service/tasks")
@RequiredArgsConstructor
public class TaskController {

    private final TaskCommandService taskCommandService;
    private final TaskQueryService taskQueryService;

    @PostMapping
    public Result<Task> create(@RequestBody Task task) {
        Task created = taskCommandService.create(task);
        return Result.success(created);
    }

    @PutMapping
    public Result<Task> update(@RequestBody Task task) {
        Task updated = taskCommandService.update(task);
        return Result.success(updated);
    }

    @DeleteMapping("/{id}")
    public Result<Void> delete(@PathVariable Long id) {
        taskCommandService.delete(id);
        return Result.success(null);
    }

    @GetMapping("/{id}")
    public Result<Task> getById(@PathVariable Long id) {
        Task task = taskQueryService.findById(id);
        return Result.success(task);
    }

    @GetMapping("/project/{projectId}")
    public Result<List<Task>> getByProjectId(@PathVariable Long projectId) {
        List<Task> tasks = taskQueryService.findByProjectId(projectId);
        return Result.success(tasks);
    }

    @GetMapping("/parent/{parentId}")
    public Result<List<Task>> getByParentId(@PathVariable Long parentId) {
        List<Task> tasks = taskQueryService.findByParentId(parentId);
        return Result.success(tasks);
    }

    @GetMapping("/page")
    public Result<Result.PageResult<Task>> getPage(
            @RequestParam(defaultValue = "1") int pageNum,
            @RequestParam(defaultValue = "10") int pageSize,
            @RequestParam(required = false) Long projectId,
            @RequestParam(required = false) String status,
            @RequestParam(required = false) Long assigneeId) {
        var page = taskQueryService.findPage(pageNum, pageSize, projectId, status, assigneeId);
        return Result.successPage(page.getRecords(), page.getTotal());
    }

    @PutMapping("/{id}/status")
    public Result<Task> updateStatus(@PathVariable Long id, @RequestParam String status) {
        Task updated = taskCommandService.updateStatus(id, status);
        return Result.success(updated);
    }

    @PutMapping("/{id}/progress")
    public Result<Task> updateProgress(@PathVariable Long id, @RequestParam Integer progress) {
        Task updated = taskCommandService.updateProgress(id, progress);
        return Result.success(updated);
    }
}