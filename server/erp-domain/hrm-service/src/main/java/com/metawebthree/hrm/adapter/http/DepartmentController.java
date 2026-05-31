package com.metawebthree.hrm.adapter.http;

import com.metawebthree.hrm.adapter.vo.Result;
import com.metawebthree.hrm.application.command.department.DepartmentCommandService;
import com.metawebthree.hrm.application.query.department.DepartmentQueryService;
import com.metawebthree.hrm.domain.entity.department.Department;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/hrm/department")
@RequiredArgsConstructor
public class DepartmentController {

    private final DepartmentQueryService departmentQueryService;
    private final DepartmentCommandService departmentCommandService;

    @GetMapping("/{id}")
    public Result<Department> getById(@PathVariable Long id) {
        return Result.success(departmentQueryService.getById(id));
    }

    @GetMapping("/list")
    public Result<List<Department>> listAll() {
        return Result.success(departmentQueryService.listAll());
    }

    @GetMapping("/tree")
    public Result<List<Department>> getTree() {
        return Result.success(departmentQueryService.getDepartmentTree());
    }

    @GetMapping("/children/{parentId}")
    public Result<List<Department>> listByParentId(@PathVariable Long parentId) {
        return Result.success(departmentQueryService.listByParentId(parentId));
    }

    @GetMapping("/level/{level}")
    public Result<List<Department>> listByLevel(@PathVariable Integer level) {
        return Result.success(departmentQueryService.listByLevel(level));
    }

    @GetMapping("/code/{code}")
    public Result<Department> getByCode(@PathVariable String code) {
        return Result.success(departmentQueryService.getByCode(code));
    }

    @PostMapping
    public Result<Department> create(@RequestBody Department department) {
        return Result.success(departmentCommandService.create(department));
    }

    @PutMapping
    public Result<Department> update(@RequestBody Department department) {
        return Result.success(departmentCommandService.update(department));
    }

    @DeleteMapping("/{id}")
    public Result<Void> delete(@PathVariable Long id) {
        departmentCommandService.delete(id);
        return Result.success();
    }
}