package com.metawebthree.hrm.application.query.department;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.hrm.domain.entity.department.Department;
import com.metawebthree.hrm.domain.repository.department.DepartmentRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
public class DepartmentQueryService {

    private final DepartmentRepository departmentRepository;

    public Department getById(Long id) {
        return departmentRepository.selectById(id);
    }

    public List<Department> listAll() {
        return departmentRepository.selectList(null);
    }

    public List<Department> listByParentId(Long parentId) {
        LambdaQueryWrapper<Department> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(Department::getParentId, parentId)
               .orderByAsc(Department::getSortOrder);
        return departmentRepository.selectList(wrapper);
    }

    public List<Department> listByLevel(Integer level) {
        LambdaQueryWrapper<Department> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(Department::getLevel, level)
               .orderByAsc(Department::getSortOrder);
        return departmentRepository.selectList(wrapper);
    }

    public Department getByCode(String code) {
        LambdaQueryWrapper<Department> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(Department::getCode, code);
        return departmentRepository.selectOne(wrapper);
    }

    public List<Department> getDepartmentTree() {
        List<Department> rootDepartments = listByParentId(0L);
        buildDepartmentTree(rootDepartments);
        return rootDepartments;
    }

    private void buildDepartmentTree(List<Department> departments) {
        for (Department dept : departments) {
            List<Department> children = listByParentId(dept.getId());
            if (!children.isEmpty()) {
                buildDepartmentTree(children);
            }
        }
    }
}