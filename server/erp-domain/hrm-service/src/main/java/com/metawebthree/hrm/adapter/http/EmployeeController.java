package com.metawebthree.hrm.adapter.http;

import com.metawebthree.hrm.adapter.vo.Result;
import com.metawebthree.hrm.application.command.employee.EmployeeCommandService;
import com.metawebthree.hrm.application.query.employee.EmployeeQueryService;
import com.metawebthree.hrm.domain.entity.employee.Employee;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/hrm/employee")
@RequiredArgsConstructor
public class EmployeeController {

    private final EmployeeQueryService employeeQueryService;
    private final EmployeeCommandService employeeCommandService;

    @GetMapping("/{id}")
    public Result<Employee> getById(@PathVariable Long id) {
        return Result.success(employeeQueryService.getById(id));
    }

    @GetMapping("/list")
    public Result<List<Employee>> listAll() {
        return Result.success(employeeQueryService.listAll());
    }

    @GetMapping("/department/{departmentId}")
    public Result<List<Employee>> listByDepartmentId(@PathVariable Long departmentId) {
        return Result.success(employeeQueryService.listByDepartmentId(departmentId));
    }

    @GetMapping("/position/{positionId}")
    public Result<List<Employee>> listByPositionId(@PathVariable Long positionId) {
        return Result.success(employeeQueryService.listByPositionId(positionId));
    }

    @GetMapping("/status/{status}")
    public Result<List<Employee>> listByStatus(@PathVariable Integer status) {
        return Result.success(employeeQueryService.listByStatus(status));
    }

    @GetMapping("/no/{employeeNo}")
    public Result<Employee> getByEmployeeNo(@PathVariable String employeeNo) {
        return Result.success(employeeQueryService.getByEmployeeNo(employeeNo));
    }

    @GetMapping("/search")
    public Result<List<Employee>> search(@RequestParam String keywords) {
        return Result.success(employeeQueryService.listByKeywords(keywords));
    }

    @GetMapping("/formal")
    public Result<List<Employee>> listFormalEmployees() {
        return Result.success(employeeQueryService.listFormalEmployees());
    }

    @GetMapping("/probation")
    public Result<List<Employee>> listProbationEmployees() {
        return Result.success(employeeQueryService.listProbationEmployees());
    }

    @PostMapping
    public Result<Employee> create(@RequestBody Employee employee) {
        return Result.success(employeeCommandService.create(employee));
    }

    @PutMapping
    public Result<Employee> update(@RequestBody Employee employee) {
        return Result.success(employeeCommandService.update(employee));
    }

    @DeleteMapping("/{id}")
    public Result<Void> delete(@PathVariable Long id) {
        employeeCommandService.delete(id);
        return Result.success();
    }

    @PostMapping("/transfer")
    public Result<Void> transfer(@RequestParam Long employeeId,
                                  @RequestParam Long newDepartmentId,
                                  @RequestParam Long newPositionId) {
        employeeCommandService.transfer(employeeId, newDepartmentId, newPositionId);
        return Result.success();
    }

    @PostMapping("/resign")
    public Result<Void> resign(@RequestParam Long employeeId, @RequestParam String reason) {
        employeeCommandService.resign(employeeId, reason);
        return Result.success();
    }
}