package com.metawebthree.hrm.domain.repository.employee;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.hrm.domain.entity.employee.Employee;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface EmployeeRepository extends BaseMapper<Employee> {
}