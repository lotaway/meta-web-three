package com.metawebthree.project.application.command.task;

import com.metawebthree.project.domain.entity.Task;
import com.metawebthree.project.domain.exception.TaskNotFoundException;
import com.metawebthree.project.domain.repository.task.TaskRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import java.time.LocalDateTime;

@Service
@RequiredArgsConstructor
public class TaskCommandService {

    private final TaskRepository taskRepository;

    @Transactional
    public Task create(Task task) {
        task.setStatus("PENDING");
        task.setProgress(0);
        task.setActualHours(0);
        task.setCreatedAt(LocalDateTime.now());
        task.setUpdatedAt(LocalDateTime.now());
        return taskRepository.save(task);
    }

    @Transactional
    public Task update(Task task) {
        Task existing = taskRepository.findById(task.getId());
        if (existing == null) {
            throw new TaskNotFoundException("Task not found: " + task.getId());
        }
        task.setTaskCode(existing.getTaskCode());
        task.setProjectId(existing.getProjectId());
        task.setCreatedAt(existing.getCreatedAt());
        task.setCreatedBy(existing.getCreatedBy());
        task.setUpdatedAt(LocalDateTime.now());
        return taskRepository.update(task);
    }

    @Transactional
    public void delete(Long id) {
        Task existing = taskRepository.findById(id);
        if (existing == null) {
            throw new TaskNotFoundException("Task not found: " + id);
        }
        taskRepository.delete(id);
    }

    @Transactional
    public Task updateStatus(Long id, String status) {
        Task task = taskRepository.findById(id);
        if (task == null) {
            throw new TaskNotFoundException("Task not found: " + id);
        }
        task.setStatus(status);
        task.setUpdatedAt(LocalDateTime.now());
        return taskRepository.update(task);
    }

    @Transactional
    public Task updateProgress(Long id, Integer progress) {
        Task task = taskRepository.findById(id);
        if (task == null) {
            throw new TaskNotFoundException("Task not found: " + id);
        }
        task.setProgress(progress);
        task.setUpdatedAt(LocalDateTime.now());
        return taskRepository.update(task);
    }

    @Transactional
    public Task updateActualHours(Long id, Integer actualHours) {
        Task task = taskRepository.findById(id);
        if (task == null) {
            throw new TaskNotFoundException("Task not found: " + id);
        }
        task.setActualHours(actualHours);
        task.setUpdatedAt(LocalDateTime.now());
        return taskRepository.update(task);
    }
}