package com.metawebthree.hrm.application.query.employee;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.hrm.domain.entity.employee.Employee;
import com.metawebthree.hrm.domain.repository.employee.EmployeeRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
public class EmployeeQueryService {

    private final EmployeeRepository employeeRepository;

    public Employee getById(Long id) {
        return employeeRepository.selectById(id);
    }

    public List<Employee> listAll() {
        return employeeRepository.selectList(null);
    }

    public List<Employee> listByDepartmentId(Long departmentId) {
        LambdaQueryWrapper<Employee> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(Employee::getDepartmentId, departmentId)
               .orderByAsc(Employee::getEmployeeNo);
        return employeeRepository.selectList(wrapper);
    }

    public List<Employee> listByPositionId(Long positionId) {
        LambdaQueryWrapper<Employee> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(Employee::getPositionId, positionId);
        return employeeRepository.selectList(wrapper);
    }

    public List<Employee> listByStatus(Integer status) {
        LambdaQueryWrapper<Employee> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(Employee::getStatus, status)
               .orderByAsc(Employee::getEmployeeNo);
        return employeeRepository.selectList(wrapper);
    }

    public Employee getByEmployeeNo(String employeeNo) {
        LambdaQueryWrapper<Employee> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(Employee::getEmployeeNo, employeeNo);
        return employeeRepository.selectOne(wrapper);
    }

    public Employee getByIdCard(String idCard) {
        LambdaQueryWrapper<Employee> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(Employee::getIdCard, idCard);
        return employeeRepository.selectOne(wrapper);
    }

    public List<Employee> listByKeywords(String keywords) {
        LambdaQueryWrapper<Employee> wrapper = new LambdaQueryWrapper<>();
        wrapper.and(w -> w.like(Employee::getName, keywords)
                         .or().like(Employee::getEmployeeNo, keywords)
                         .or().like(Employee::getMobile, keywords)
                         .or().like(Employee::getEmail, keywords))
               .orderByAsc(Employee::getEmployeeNo);
        return employeeRepository.selectList(wrapper);
    }

    public List<Employee> listFormalEmployees() {
        return listByStatus(1);
    }

    public List<Employee> listProbationEmployees() {
        return listByStatus(0);
    }
}