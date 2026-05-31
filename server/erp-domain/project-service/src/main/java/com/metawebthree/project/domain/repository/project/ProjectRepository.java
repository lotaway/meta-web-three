package com.metawebthree.project.domain.repository.project;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.project.domain.entity.Project;
import java.util.List;

public interface ProjectRepository {
    Project save(Project project);
    Project update(Project project);
    void delete(Long id);
    Project findById(Long id);
    Project findByCode(String projectCode);
    List<Project> findAll();
    IPage<Project> findPage(Page<Project> page, String keyword, String status, Long departmentId);
}