package com.metawebthree.cs.infrastructure.persistence.mybatis;

import com.metawebthree.cs.domain.model.WorkOrder;
import com.metawebthree.cs.domain.model.enums.WorkOrderCategory;
import com.metawebthree.cs.domain.model.enums.WorkOrderStatus;
import com.metawebthree.cs.domain.repository.WorkOrderRepository;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
@RequiredArgsConstructor
public class MybatisWorkOrderRepository implements WorkOrderRepository {

    private final MybatisWorkOrderMapper workOrderMapper;

    @Override
    public WorkOrder save(WorkOrder workOrder) {
        if (workOrder.getId() == null) {
            workOrderMapper.insert(workOrder);
        } else {
            workOrder.setUpdateTime(java.time.LocalDateTime.now());
            workOrderMapper.updateById(workOrder);
        }
        return workOrder;
    }

    @Override
    public WorkOrder findById(Long id) {
        return workOrderMapper.selectById(id);
    }

    @Override
    public List<WorkOrder> findByCustomerId(Long customerId) {
        return workOrderMapper.findByCustomerId(customerId);
    }

    @Override
    public List<WorkOrder> findByAgentId(Long agentId) {
        return workOrderMapper.findByAgentId(agentId);
    }

    @Override
    public List<WorkOrder> findByStatus(WorkOrderStatus status) {
        return workOrderMapper.findByStatus(status);
    }

    @Override
    public List<WorkOrder> findByCategory(WorkOrderCategory category) {
        return workOrderMapper.findByCategory(category);
    }

    @Override
    public List<WorkOrder> findPending() {
        return workOrderMapper.findPending();
    }

    @Override
    public void deleteById(Long id) {
        workOrderMapper.deleteById(id);
    }

    @Override
    public List<WorkOrder> findAll() {
        return workOrderMapper.selectList(new LambdaQueryWrapper<WorkOrder>()
                .orderByDesc(WorkOrder::getPriority)
                .orderByDesc(WorkOrder::getCreateTime));
    }

    @Override
    public Long countByStatus(WorkOrderStatus status) {
        return workOrderMapper.countByStatus(status);
    }

    @Override
    public Long countByCategory(WorkOrderCategory category) {
        return workOrderMapper.countByCategory(category);
    }
}