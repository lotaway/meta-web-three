package com.metawebthree.hrm.domain.repository.department;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.metawebthree.hrm.domain.entity.department.Department;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface DepartmentRepository extends BaseMapper<Department> {
}