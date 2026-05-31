package com.metawebthree.project.infrastructure.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.project.domain.entity.Task;
import com.metawebthree.project.domain.repository.task.TaskRepository;
import com.metawebthree.project.infrastructure.mapper.TaskMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Repository;
import java.util.List;

@Repository
@RequiredArgsConstructor
public class TaskRepositoryImpl implements TaskRepository {

    private final TaskMapper taskMapper;

    @Override
    public Task save(Task task) {
        taskMapper.insert(task);
        return task;
    }

    @Override
    public Task update(Task task) {
        taskMapper.updateById(task);
        return task;
    }

    @Override
    public void delete(Long id) {
        taskMapper.deleteById(id);
    }

    @Override
    public Task findById(Long id) {
        return taskMapper.selectById(id);
    }

    @Override
    public List<Task> findByProjectId(Long projectId) {
        LambdaQueryWrapper<Task> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(Task::getProjectId, projectId).orderByAsc(Task::getSort);
        return taskMapper.selectList(wrapper);
    }

    @Override
    public List<Task> findByParentId(Long parentId) {
        LambdaQueryWrapper<Task> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(Task::getParentId, parentId).orderByAsc(Task::getSort);
        return taskMapper.selectList(wrapper);
    }

    @Override
    public IPage<Task> findPage(Page<Task> page, Long projectId, String status, Long assigneeId) {
        LambdaQueryWrapper<Task> wrapper = new LambdaQueryWrapper<>();
        if (projectId != null) {
            wrapper.eq(Task::getProjectId, projectId);
        }
        if (status != null && !status.isEmpty()) {
            wrapper.eq(Task::getStatus, status);
        }
        if (assigneeId != null) {
            wrapper.eq(Task::getAssigneeId, assigneeId);
        }
        wrapper.orderByDesc(Task::getCreatedAt);
        return taskMapper.selectPage(page, wrapper);
    }
}