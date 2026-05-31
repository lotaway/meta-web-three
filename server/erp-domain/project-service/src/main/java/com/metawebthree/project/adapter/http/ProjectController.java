package com.metawebthree.project.adapter.http;

import com.metawebthree.project.adapter.vo.Result;
import com.metawebthree.project.application.command.project.ProjectCommandService;
import com.metawebthree.project.application.query.project.ProjectQueryService;
import com.metawebthree.project.domain.entity.Project;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@RestController
@RequestMapping("/project-service/projects")
@RequiredArgsConstructor
public class ProjectController {

    private final ProjectCommandService projectCommandService;
    private final ProjectQueryService projectQueryService;

    @PostMapping
    public Result<Project> create(@RequestBody Project project) {
        Project created = projectCommandService.create(project);
        return Result.success(created);
    }

    @PutMapping
    public Result<Project> update(@RequestBody Project project) {
        Project updated = projectCommandService.update(project);
        return Result.success(updated);
    }

    @DeleteMapping("/{id}")
    public Result<Void> delete(@PathVariable Long id) {
        projectCommandService.delete(id);
        return Result.success(null);
    }

    @GetMapping("/{id}")
    public Result<Project> getById(@PathVariable Long id) {
        Project project = projectQueryService.findById(id);
        return Result.success(project);
    }

    @GetMapping("/code/{projectCode}")
    public Result<Project> getByCode(@PathVariable String projectCode) {
        Project project = projectQueryService.findByCode(projectCode);
        return Result.success(project);
    }

    @GetMapping
    public Result<List<Project>> getAll() {
        List<Project> projects = projectQueryService.findAll();
        return Result.success(projects);
    }

    @GetMapping("/page")
    public Result<Result.PageResult<Project>> getPage(
            @RequestParam(defaultValue = "1") int pageNum,
            @RequestParam(defaultValue = "10") int pageSize,
            @RequestParam(required = false) String keyword,
            @RequestParam(required = false) String status,
            @RequestParam(required = false) Long departmentId) {
        var page = projectQueryService.findPage(pageNum, pageSize, keyword, status, departmentId);
        return Result.successPage(page.getRecords(), page.getTotal());
    }

    @PutMapping("/{id}/status")
    public Result<Project> updateStatus(@PathVariable Long id, @RequestParam String status) {
        Project updated = projectCommandService.updateStatus(id, status);
        return Result.success(updated);
    }

    @PutMapping("/{id}/progress")
    public Result<Project> updateProgress(@PathVariable Long id, @RequestParam Integer progress) {
        Project updated = projectCommandService.updateProgress(id, progress);
        return Result.success(updated);
    }
}