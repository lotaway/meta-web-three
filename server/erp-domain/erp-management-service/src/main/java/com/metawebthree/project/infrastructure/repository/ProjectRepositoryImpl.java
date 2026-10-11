package com.metawebthree.project.infrastructure.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.project.domain.entity.Project;
import com.metawebthree.project.domain.repository.project.ProjectRepository;
import com.metawebthree.project.infrastructure.mapper.ProjectMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Repository;
import java.util.List;

@Repository
@RequiredArgsConstructor
public class ProjectRepositoryImpl implements ProjectRepository {

    private final ProjectMapper projectMapper;

    @Override
    public Project save(Project project) {
        projectMapper.insert(project);
        return project;
    }

    @Override
    public Project update(Project project) {
        projectMapper.updateById(project);
        return project;
    }

    @Override
    public void delete(Long id) {
        projectMapper.deleteById(id);
    }

    @Override
    public Project findById(Long id) {
        return projectMapper.selectById(id);
    }

    @Override
    public Project findByCode(String projectCode) {
        LambdaQueryWrapper<Project> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(Project::getProjectCode, projectCode);
        return projectMapper.selectOne(wrapper);
    }

    @Override
    public List<Project> findAll() {
        return projectMapper.selectList(null);
    }

    @Override
    public IPage<Project> findPage(Page<Project> page, String keyword, String status, Long departmentId) {
        LambdaQueryWrapper<Project> wrapper = new LambdaQueryWrapper<>();
        if (keyword != null && !keyword.isEmpty()) {
            wrapper.like(Project::getProjectName, keyword)
                   .or().like(Project::getProjectCode, keyword);
        }
        if (status != null && !status.isEmpty()) {
            wrapper.eq(Project::getStatus, status);
        }
        if (departmentId != null) {
            wrapper.eq(Project::getDepartmentId, departmentId);
        }
        wrapper.orderByDesc(Project::getCreatedAt);
        return projectMapper.selectPage(page, wrapper);
    }
}