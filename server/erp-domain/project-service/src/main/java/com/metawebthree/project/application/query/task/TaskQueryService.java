package com.metawebthree.project.application.query.task;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.project.domain.entity.Task;
import com.metawebthree.project.domain.repository.task.TaskRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import java.util.List;

@Service
@RequiredArgsConstructor
public class TaskQueryService {

    private final TaskRepository taskRepository;

    public Task findById(Long id) {
        return taskRepository.findById(id);
    }

    public List<Task> findByProjectId(Long projectId) {
        return taskRepository.findByProjectId(projectId);
    }

    public List<Task> findByParentId(Long parentId) {
        return taskRepository.findByParentId(parentId);
    }

    public IPage<Task> findPage(int pageNum, int pageSize, Long projectId, String status, Long assigneeId) {
        Page<Task> page = new Page<>(pageNum, pageSize);
        return taskRepository.findPage(page, projectId, status, assigneeId);
    }
}