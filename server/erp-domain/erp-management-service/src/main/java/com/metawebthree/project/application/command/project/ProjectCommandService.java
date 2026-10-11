package com.metawebthree.project.application.command.project;

import com.metawebthree.project.domain.entity.Project;
import com.metawebthree.project.domain.exception.ProjectNotFoundException;
import com.metawebthree.project.domain.repository.project.ProjectRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import java.time.LocalDateTime;

@Service
@RequiredArgsConstructor
public class ProjectCommandService {

    private final ProjectRepository projectRepository;

    @Transactional
    public Project create(Project project) {
        project.setStatus("DRAFT");
        project.setUsedAmount(java.math.BigDecimal.ZERO);
        project.setProgress(0);
        project.setCreatedAt(LocalDateTime.now());
        project.setUpdatedAt(LocalDateTime.now());
        return projectRepository.save(project);
    }

    @Transactional
    public Project update(Project project) {
        Project existing = projectRepository.findById(project.getId());
        if (existing == null) {
            throw new ProjectNotFoundException("Project not found: " + project.getId());
        }
        project.setProjectCode(existing.getProjectCode());
        project.setCreatedAt(existing.getCreatedAt());
        project.setCreatedBy(existing.getCreatedBy());
        project.setUpdatedAt(LocalDateTime.now());
        return projectRepository.update(project);
    }

    @Transactional
    public void delete(Long id) {
        Project existing = projectRepository.findById(id);
        if (existing == null) {
            throw new ProjectNotFoundException("Project not found: " + id);
        }
        projectRepository.delete(id);
    }

    @Transactional
    public Project updateStatus(Long id, String status) {
        Project project = projectRepository.findById(id);
        if (project == null) {
            throw new ProjectNotFoundException("Project not found: " + id);
        }
        project.setStatus(status);
        project.setUpdatedAt(LocalDateTime.now());
        return projectRepository.update(project);
    }

    @Transactional
    public Project updateProgress(Long id, Integer progress) {
        Project project = projectRepository.findById(id);
        if (project == null) {
            throw new ProjectNotFoundException("Project not found: " + id);
        }
        project.setProgress(progress);
        project.setUpdatedAt(LocalDateTime.now());
        return projectRepository.update(project);
    }
}