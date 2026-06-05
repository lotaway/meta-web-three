package com.metawebthree.mes.infrastructure.persistence.repository.scheduling;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.mes.domain.entity.scheduling.ScheduleOrder;
import com.metawebthree.mes.domain.entity.scheduling.ScheduleOrder.ScheduleOperation;
import com.metawebthree.mes.domain.entity.scheduling.ScheduleOrder.ScheduleOperationStatus;
import com.metawebthree.mes.domain.entity.scheduling.ScheduleOrder.ScheduleStatus;
import com.metawebthree.mes.domain.entity.scheduling.ScheduleOrder.Priority;
import com.metawebthree.mes.domain.repository.scheduling.ScheduleOrderRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.scheduling.ScheduleOperationDO;
import com.metawebthree.mes.infrastructure.persistence.dataobject.scheduling.ScheduleOrderDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.scheduling.ScheduleOperationMapper;
import com.metawebthree.mes.infrastructure.persistence.mapper.scheduling.ScheduleOrderMapper;
import org.springframework.stereotype.Repository;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class ScheduleOrderRepositoryImpl implements ScheduleOrderRepository {

    private final ScheduleOrderMapper orderMapper;
    private final ScheduleOperationMapper operationMapper;

    public ScheduleOrderRepositoryImpl(ScheduleOrderMapper orderMapper,
                                       ScheduleOperationMapper operationMapper) {
        this.orderMapper = orderMapper;
        this.operationMapper = operationMapper;
    }

    @Override
    public Optional<ScheduleOrder> findById(Long id) {
        ScheduleOrderDO doObj = orderMapper.selectById(id);
        return Optional.ofNullable(doObj).map(this::toEntityWithOperations);
    }

    @Override
    public Optional<ScheduleOrder> findByScheduleNo(String scheduleNo) {
        LambdaQueryWrapper<ScheduleOrderDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ScheduleOrderDO::getScheduleNo, scheduleNo);
        ScheduleOrderDO doObj = orderMapper.selectOne(wrapper);
        return Optional.ofNullable(doObj).map(this::toEntityWithOperations);
    }

    @Override
    public List<ScheduleOrder> findByStatus(ScheduleOrder.ScheduleStatus status) {
        LambdaQueryWrapper<ScheduleOrderDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ScheduleOrderDO::getStatus, status.name());
        List<ScheduleOrderDO> doList = orderMapper.selectList(wrapper);
        return doList.stream().map(this::toEntityWithOperations).collect(Collectors.toList());
    }

    @Override
    public List<ScheduleOrder> findByWorkshopId(String workshopId) {
        LambdaQueryWrapper<ScheduleOrderDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ScheduleOrderDO::getWorkshopId, workshopId);
        List<ScheduleOrderDO> doList = orderMapper.selectList(wrapper);
        return doList.stream().map(this::toEntityWithOperations).collect(Collectors.toList());
    }

    @Override
    public List<ScheduleOrder> findByDueDateBetween(LocalDateTime start, LocalDateTime end) {
        LambdaQueryWrapper<ScheduleOrderDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.between(ScheduleOrderDO::getDueDate, start, end);
        List<ScheduleOrderDO> doList = orderMapper.selectList(wrapper);
        return doList.stream().map(this::toEntityWithOperations).collect(Collectors.toList());
    }

    @Override
    public List<ScheduleOrder> findOverdueOrders() {
        LambdaQueryWrapper<ScheduleOrderDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.lt(ScheduleOrderDO::getDueDate, LocalDateTime.now());
        wrapper.ne(ScheduleOrderDO::getStatus, ScheduleStatus.COMPLETED.name());
        wrapper.ne(ScheduleOrderDO::getStatus, ScheduleStatus.CANCELLED.name());
        List<ScheduleOrderDO> doList = orderMapper.selectList(wrapper);
        return doList.stream().map(this::toEntityWithOperations).collect(Collectors.toList());
    }

    @Override
    public List<ScheduleOrder> findAll() {
        List<ScheduleOrderDO> doList = orderMapper.selectList(null);
        return doList.stream().map(this::toEntityWithOperations).collect(Collectors.toList());
    }

    @Override
    public ScheduleOrder save(ScheduleOrder entity) {
        ScheduleOrderDO doObj = toDO(entity);
        if (doObj.getId() == null) {
            orderMapper.insert(doObj);
            entity.setId(doObj.getId());
        } else {
            orderMapper.updateById(doObj);
        }
        saveOperations(entity.getId(), entity.getOperations());
        return entity;
    }

    @Override
    public void update(ScheduleOrder entity) {
        if (entity.getId() != null) {
            ScheduleOrderDO doObj = toDO(entity);
            orderMapper.updateById(doObj);
            saveOperations(entity.getId(), entity.getOperations());
        }
    }

    @Override
    public void deleteById(Long id) {
        LambdaQueryWrapper<ScheduleOperationDO> opWrapper = new LambdaQueryWrapper<>();
        opWrapper.eq(ScheduleOperationDO::getScheduleOrderId, id);
        operationMapper.delete(opWrapper);
        orderMapper.deleteById(id);
    }

    private void saveOperations(Long orderId, List<ScheduleOperation> operations) {
        LambdaQueryWrapper<ScheduleOperationDO> opWrapper = new LambdaQueryWrapper<>();
        opWrapper.eq(ScheduleOperationDO::getScheduleOrderId, orderId);
        operationMapper.delete(opWrapper);
        if (operations != null && !operations.isEmpty()) {
            for (ScheduleOperation op : operations) {
                operationMapper.insert(toOperationDO(orderId, op));
            }
        }
    }

    private ScheduleOrder toEntityWithOperations(ScheduleOrderDO doObj) {
        ScheduleOrder entity = toEntity(doObj);
        LambdaQueryWrapper<ScheduleOperationDO> opWrapper = new LambdaQueryWrapper<>();
        opWrapper.eq(ScheduleOperationDO::getScheduleOrderId, doObj.getId());
        opWrapper.orderByAsc(ScheduleOperationDO::getSequenceNo);
        List<ScheduleOperationDO> opDOs = operationMapper.selectList(opWrapper);
        List<ScheduleOperation> ops = opDOs.stream().map(this::toOperation).collect(Collectors.toList());
        entity.setOperations(ops);
        return entity;
    }

    private ScheduleOrder toEntity(ScheduleOrderDO doObj) {
        if (doObj == null) return null;
        ScheduleOrder entity = new ScheduleOrder();
        entity.setId(doObj.getId());
        entity.setScheduleNo(doObj.getScheduleNo());
        entity.setOrderNo(doObj.getOrderNo());
        entity.setProductCode(doObj.getProductCode());
        entity.setProductName(doObj.getProductName());
        entity.setQuantity(doObj.getQuantity());
        entity.setCompletedQuantity(doObj.getCompletedQuantity() != null ? doObj.getCompletedQuantity() : BigDecimal.ZERO);
        entity.setDueDate(doObj.getDueDate());
        entity.setScheduledStartTime(doObj.getScheduledStartTime());
        entity.setScheduledEndTime(doObj.getScheduledEndTime());
        entity.setActualStartTime(doObj.getActualStartTime());
        entity.setActualEndTime(doObj.getActualEndTime());
        entity.setPriority(doObj.getPriority() != null ? Priority.valueOf(doObj.getPriority()) : null);
        entity.setStatus(doObj.getStatus() != null ? ScheduleStatus.valueOf(doObj.getStatus()) : ScheduleStatus.PENDING);
        entity.setWorkshopId(doObj.getWorkshopId());
        entity.setRouteCode(doObj.getRouteCode());
        entity.setRemark(doObj.getRemark());
        entity.setCreatedBy(doObj.getCreatedBy());
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        return entity;
    }

    private ScheduleOrderDO toDO(ScheduleOrder entity) {
        if (entity == null) return null;
        ScheduleOrderDO doObj = new ScheduleOrderDO();
        doObj.setId(entity.getId());
        doObj.setScheduleNo(entity.getScheduleNo());
        doObj.setOrderNo(entity.getOrderNo());
        doObj.setProductCode(entity.getProductCode());
        doObj.setProductName(entity.getProductName());
        doObj.setQuantity(entity.getQuantity());
        doObj.setCompletedQuantity(entity.getCompletedQuantity());
        doObj.setDueDate(entity.getDueDate());
        doObj.setScheduledStartTime(entity.getScheduledStartTime());
        doObj.setScheduledEndTime(entity.getScheduledEndTime());
        doObj.setActualStartTime(entity.getActualStartTime());
        doObj.setActualEndTime(entity.getActualEndTime());
        doObj.setPriority(entity.getPriority() != null ? entity.getPriority().name() : null);
        doObj.setStatus(entity.getStatus() != null ? entity.getStatus().name() : ScheduleStatus.PENDING.name());
        doObj.setWorkshopId(entity.getWorkshopId());
        doObj.setRouteCode(entity.getRouteCode());
        doObj.setRemark(entity.getRemark());
        doObj.setCreatedBy(entity.getCreatedBy());
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        return doObj;
    }

    private ScheduleOperation toOperation(ScheduleOperationDO doObj) {
        if (doObj == null) return null;
        ScheduleOperation op = new ScheduleOperation();
        op.setId(doObj.getId());
        op.setScheduleOrderId(doObj.getScheduleOrderId());
        op.setOperationCode(doObj.getOperationCode());
        op.setOperationName(doObj.getOperationName());
        op.setSequenceNo(doObj.getSequenceNo());
        op.setResourceCode(doObj.getResourceCode());
        op.setResourceName(doObj.getResourceName());
        op.setSetupTimeMinutes(doObj.getSetupTimeMinutes());
        op.setProcessingTimeMinutes(doObj.getProcessingTimeMinutes());
        op.setTeardownTimeMinutes(doObj.getTeardownTimeMinutes());
        op.setStatus(doObj.getStatus() != null ? ScheduleOperationStatus.valueOf(doObj.getStatus()) : ScheduleOperationStatus.PENDING);
        op.setScheduledStartTime(doObj.getScheduledStartTime());
        op.setScheduledEndTime(doObj.getScheduledEndTime());
        return op;
    }

    private ScheduleOperationDO toOperationDO(Long orderId, ScheduleOperation entity) {
        if (entity == null) return null;
        ScheduleOperationDO doObj = new ScheduleOperationDO();
        doObj.setId(entity.getId());
        doObj.setScheduleOrderId(orderId);
        doObj.setOperationCode(entity.getOperationCode());
        doObj.setOperationName(entity.getOperationName());
        doObj.setSequenceNo(entity.getSequenceNo());
        doObj.setResourceCode(entity.getResourceCode());
        doObj.setResourceName(entity.getResourceName());
        doObj.setSetupTimeMinutes(entity.getSetupTimeMinutes());
        doObj.setProcessingTimeMinutes(entity.getProcessingTimeMinutes());
        doObj.setTeardownTimeMinutes(entity.getTeardownTimeMinutes());
        doObj.setStatus(entity.getStatus() != null ? entity.getStatus().name() : ScheduleOperationStatus.PENDING.name());
        doObj.setScheduledStartTime(entity.getScheduledStartTime());
        doObj.setScheduledEndTime(entity.getScheduledEndTime());
        return doObj;
    }
}
