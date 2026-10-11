package com.metawebthree.hrm.application.command.department;

import com.metawebthree.hrm.domain.entity.department.Department;
import com.metawebthree.hrm.domain.exception.DepartmentNotFoundException;
import com.metawebthree.hrm.domain.repository.department.DepartmentRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
public class DepartmentCommandService {

    private final DepartmentRepository departmentRepository;

    @Transactional
    public Department create(Department department) {
        if (department.getParentId() != null && department.getParentId() > 0) {
            Department parent = departmentRepository.selectById(department.getParentId());
            if (parent != null) {
                department.setLevel(parent.getLevel() + 1);
            }
        } else {
            department.setLevel(1);
            department.setParentId(0L);
        }
        departmentRepository.insert(department);
        return department;
    }

    @Transactional
    public Department update(Department department) {
        Department existing = departmentRepository.selectById(department.getId());
        if (existing == null) {
            throw new DepartmentNotFoundException(department.getId());
        }
        
        if (!existing.getParentId().equals(department.getParentId())) {
            if (department.getParentId() != null && department.getParentId() > 0) {
                Department parent = departmentRepository.selectById(department.getParentId());
                if (parent != null) {
                    department.setLevel(parent.getLevel() + 1);
                }
            } else {
                department.setLevel(1);
            }
        }
        
        departmentRepository.updateById(department);
        return department;
    }

    @Transactional
    public void delete(Long id) {
        departmentRepository.deleteById(id);
    }
}