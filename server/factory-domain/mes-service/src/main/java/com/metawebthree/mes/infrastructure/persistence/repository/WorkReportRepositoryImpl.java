package com.metawebthree.mes.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.mes.domain.entity.WorkReport;
import com.metawebthree.mes.domain.repository.WorkReportRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.WorkReportDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.WorkReportMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public class WorkReportRepositoryImpl implements WorkReportRepository {
    
    @Autowired
    private WorkReportMapper workReportMapper;
    
    @Override
    public Optional<WorkReport> findById(Long id) {
        WorkReportDO reportDO = workReportMapper.selectById(id);
        return Optional.ofNullable(reportDO).map(this::toEntity);
    }
    
    @Override
    public Optional<WorkReport> findByReportNo(String reportNo) {
        LambdaQueryWrapper<WorkReportDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(WorkReportDO::getReportNo, reportNo);
        WorkReportDO reportDO = workReportMapper.selectOne(wrapper);
        return Optional.ofNullable(reportDO).map(this::toEntity);
    }
    
    @Override
    public List<WorkReport> findByTaskId(Long taskId) {
        LambdaQueryWrapper<WorkReportDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(WorkReportDO::getTaskId, taskId);
        List<WorkReportDO> doList = workReportMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<WorkReport> findByWorkOrderId(Long workOrderId) {
        LambdaQueryWrapper<WorkReportDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(WorkReportDO::getWorkOrderId, workOrderId);
        List<WorkReportDO> doList = workReportMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<WorkReport> findByOperatorId(String operatorId) {
        LambdaQueryWrapper<WorkReportDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(WorkReportDO::getOperatorId, operatorId);
        List<WorkReportDO> doList = workReportMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<WorkReport> findByStatus(WorkReport.ReportStatus status) {
        LambdaQueryWrapper<WorkReportDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(WorkReportDO::getStatus, status.name());
        List<WorkReportDO> doList = workReportMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<WorkReport> findByWorkstationId(String workstationId) {
        LambdaQueryWrapper<WorkReportDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(WorkReportDO::getWorkstationId, workstationId);
        List<WorkReportDO> doList = workReportMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public List<WorkReport> findAll() {
        List<WorkReportDO> doList = workReportMapper.selectList(null);
        return doList.stream().map(this::toEntity).collect(java.util.stream.Collectors.toList());
    }
    
    @Override
    public WorkReport save(WorkReport report) {
        WorkReportDO reportDO = toDO(report);
        if (report.getId() == null) {
            workReportMapper.insert(reportDO);
            report.setId(reportDO.getId());
        } else {
            workReportMapper.updateById(reportDO);
        }
        return report;
    }
    
    @Override
    public void update(WorkReport report) {
        if (report.getId() != null) {
            WorkReportDO reportDO = toDO(report);
            workReportMapper.updateById(reportDO);
        }
    }
    
    @Override
    public void deleteById(Long id) {
        workReportMapper.deleteById(id);
    }
    
    private WorkReport toEntity(WorkReportDO doObj) {
        if (doObj == null) {
            return null;
        }
        WorkReport entity = new WorkReport();
        entity.setId(doObj.getId());
        entity.setReportNo(doObj.getReportNo());
        entity.setTaskId(doObj.getTaskId());
        entity.setTaskNo(doObj.getTaskNo());
        entity.setWorkOrderId(doObj.getWorkOrderId());
        entity.setWorkOrderNo(doObj.getWorkOrderNo());
        entity.setWorkstationId(doObj.getWorkstationId());
        entity.setWorkstationName(doObj.getWorkstationName());
        entity.setProcessCode(doObj.getProcessCode());
        entity.setProcessName(doObj.getProcessName());
        entity.setStepNo(doObj.getStepNo());
        entity.setOperatorId(doObj.getOperatorId());
        entity.setOperatorName(doObj.getOperatorName());
        entity.setReportTime(doObj.getReportTime());
        entity.setQuantity(doObj.getQuantity());
        entity.setQualifiedQuantity(doObj.getQualifiedQuantity());
        entity.setDefectiveQuantity(doObj.getDefectiveQuantity());
        entity.setDurationMinutes(doObj.getDurationMinutes());
        entity.setParameterValuesJson(doObj.getParameterValuesJson());
        entity.setRemarks(doObj.getRemarks());
        entity.setStatus(WorkReport.ReportStatus.valueOf(doObj.getStatus()));
        entity.setCreatedBy(doObj.getCreatedBy());
        entity.setCreatedAt(doObj.getCreatedAt());
        return entity;
    }
    
    private WorkReportDO toDO(WorkReport entity) {
        if (entity == null) {
            return null;
        }
        WorkReportDO doObj = new WorkReportDO();
        doObj.setId(entity.getId());
        doObj.setReportNo(entity.getReportNo());
        doObj.setTaskId(entity.getTaskId());
        doObj.setTaskNo(entity.getTaskNo());
        doObj.setWorkOrderId(entity.getWorkOrderId());
        doObj.setWorkOrderNo(entity.getWorkOrderNo());
        doObj.setWorkstationId(entity.getWorkstationId());
        doObj.setWorkstationName(entity.getWorkstationName());
        doObj.setProcessCode(entity.getProcessCode());
        doObj.setProcessName(entity.getProcessName());
        doObj.setStepNo(entity.getStepNo());
        doObj.setOperatorId(entity.getOperatorId());
        doObj.setOperatorName(entity.getOperatorName());
        doObj.setReportTime(entity.getReportTime());
        doObj.setQuantity(entity.getQuantity());
        doObj.setQualifiedQuantity(entity.getQualifiedQuantity());
        doObj.setDefectiveQuantity(entity.getDefectiveQuantity());
        doObj.setDurationMinutes(entity.getDurationMinutes());
        doObj.setParameterValuesJson(entity.getParameterValuesJson());
        doObj.setRemarks(entity.getRemarks());
        doObj.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        doObj.setCreatedBy(entity.getCreatedBy());
        return doObj;
    }
}