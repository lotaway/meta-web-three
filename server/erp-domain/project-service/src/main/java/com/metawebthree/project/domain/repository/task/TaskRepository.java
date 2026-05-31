package com.metawebthree.project.domain.repository.task;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.project.domain.entity.Task;
import java.util.List;

public interface TaskRepository {
    Task save(Task task);
    Task update(Task task);
    void delete(Long id);
    Task findById(Long id);
    List<Task> findByProjectId(Long projectId);
    List<Task> findByParentId(Long parentId);
    IPage<Task> findPage(Page<Task> page, Long projectId, String status, Long assigneeId);
}