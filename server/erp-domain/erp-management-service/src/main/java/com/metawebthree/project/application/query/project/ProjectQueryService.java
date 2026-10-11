package com.metawebthree.project.application.query.project;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.project.domain.entity.Project;
import com.metawebthree.project.domain.repository.project.ProjectRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import java.util.List;

@Service
@RequiredArgsConstructor
public class ProjectQueryService {

    private final ProjectRepository projectRepository;

    public Project findById(Long id) {
        return projectRepository.findById(id);
    }

    public Project findByCode(String projectCode) {
        return projectRepository.findByCode(projectCode);
    }

    public List<Project> findAll() {
        return projectRepository.findAll();
    }

    public IPage<Project> findPage(int pageNum, int pageSize, String keyword, String status, Long departmentId) {
        Page<Project> page = new Page<>(pageNum, pageSize);
        return projectRepository.findPage(page, keyword, status, departmentId);
    }
}