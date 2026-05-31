package com.metawebthree.hrm.application.command.employee;

import com.metawebthree.hrm.domain.entity.employee.Employee;
import com.metawebthree.hrm.domain.exception.EmployeeNotFoundException;
import com.metawebthree.hrm.domain.repository.employee.EmployeeRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
public class EmployeeCommandService {

    private final EmployeeRepository employeeRepository;

    @Transactional
    public Employee create(Employee employee) {
        employeeRepository.insert(employee);
        return employee;
    }

    @Transactional
    public Employee update(Employee employee) {
        Employee existing = employeeRepository.selectById(employee.getId());
        if (existing == null) {
            throw new EmployeeNotFoundException(employee.getId());
        }
        employeeRepository.updateById(employee);
        return employee;
    }

    @Transactional
    public void delete(Long id) {
        employeeRepository.deleteById(id);
    }

    @Transactional
    public void transfer(Long employeeId, Long newDepartmentId, Long newPositionId) {
        Employee employee = employeeRepository.selectById(employeeId);
        if (employee == null) {
            throw new EmployeeNotFoundException(employeeId);
        }
        employee.setDepartmentId(newDepartmentId);
        employee.setPositionId(newPositionId);
        employeeRepository.updateById(employee);
    }

    @Transactional
    public void resign(Long employeeId, String reason) {
        Employee employee = employeeRepository.selectById(employeeId);
        if (employee == null) {
            throw new EmployeeNotFoundException(employeeId);
        }
        employee.setStatus(2);
        employee.setRemark(reason);
        employeeRepository.updateById(employee);
    }
}