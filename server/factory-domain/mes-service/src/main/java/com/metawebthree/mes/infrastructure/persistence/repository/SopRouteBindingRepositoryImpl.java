package com.metawebthree.mes.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.mes.domain.entity.SopDocument;
import com.metawebthree.mes.domain.entity.SopRouteBinding;
import com.metawebthree.mes.domain.repository.SopDocumentRepository;
import com.metawebthree.mes.domain.repository.SopRouteBindingRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.SopDocumentDO;
import com.metawebthree.mes.infrastructure.persistence.dataobject.SopRouteBindingDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.SopDocumentMapper;
import com.metawebthree.mes.infrastructure.persistence.mapper.SopRouteBindingMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.stream.Collectors;

@Repository
public class SopRouteBindingRepositoryImpl implements SopRouteBindingRepository {

    @Autowired
    private SopRouteBindingMapper sopRouteBindingMapper;

    @Autowired
    private SopDocumentMapper sopDocumentMapper;

    @Autowired
    private SopDocumentRepository sopDocumentRepository;

    @Override
    public List<SopDocument> findByRouteCodeAndStepNo(String routeCode, Integer stepNo) {
        LambdaQueryWrapper<SopRouteBindingDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(SopRouteBindingDO::getRouteCode, routeCode)
               .eq(stepNo != null, SopRouteBindingDO::getStepNo, stepNo)
               .eq(SopRouteBindingDO::getIsActive, true);
        
        List<SopRouteBindingDO> bindings = sopRouteBindingMapper.selectList(wrapper);
        List<Long> docIds = bindings.stream()
            .map(SopRouteBindingDO::getSopDocumentId)
            .distinct()
            .collect(Collectors.toList());
        
        return docIds.stream()
            .map(sopDocumentRepository::findById)
            .filter(opt -> opt.isPresent())
            .map(opt -> opt.get())
            .collect(Collectors.toList());
    }

    @Override
    public List<SopDocument> findByWorkstationId(String workstationId) {
        LambdaQueryWrapper<SopRouteBindingDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(SopRouteBindingDO::getWorkstationId, workstationId)
               .eq(SopRouteBindingDO::getIsActive, true);
        
        List<SopRouteBindingDO> bindings = sopRouteBindingMapper.selectList(wrapper);
        List<Long> docIds = bindings.stream()
            .map(SopRouteBindingDO::getSopDocumentId)
            .distinct()
            .collect(Collectors.toList());
        
        return docIds.stream()
            .map(sopDocumentRepository::findById)
            .filter(opt -> opt.isPresent())
            .map(opt -> opt.get())
            .collect(Collectors.toList());
    }

    @Override
    public List<SopRouteBinding> findBySopDocumentId(Long sopDocumentId) {
        LambdaQueryWrapper<SopRouteBindingDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(SopRouteBindingDO::getSopDocumentId, sopDocumentId);
        
        List<SopRouteBindingDO> doList = sopRouteBindingMapper.selectList(wrapper);
        return doList.stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public void save(SopRouteBinding binding) {
        SopRouteBindingDO doObj = toDO(binding);
        if (binding.getId() == null) {
            sopRouteBindingMapper.insert(doObj);
            binding.setId(doObj.getId());
        } else {
            sopRouteBindingMapper.updateById(doObj);
        }
    }

    @Override
    public void deleteBySopDocumentId(Long sopDocumentId) {
        LambdaQueryWrapper<SopRouteBindingDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(SopRouteBindingDO::getSopDocumentId, sopDocumentId);
        sopRouteBindingMapper.delete(wrapper);
    }

    private SopRouteBinding toEntity(SopRouteBindingDO doObj) {
        if (doObj == null) {
            return null;
        }
        SopRouteBinding entity = new SopRouteBinding();
        entity.setId(doObj.getId());
        entity.setSopDocumentId(doObj.getSopDocumentId());
        entity.setRouteCode(doObj.getRouteCode());
        entity.setRouteName(doObj.getRouteName());
        entity.setStepNo(doObj.getStepNo());
        entity.setProcessCode(doObj.getProcessCode());
        entity.setProcessName(doObj.getProcessName());
        entity.setWorkstationId(doObj.getWorkstationId());
        entity.setWorkstationName(doObj.getWorkstationName());
        entity.setSortOrder(doObj.getSortOrder());
        entity.setIsActive(doObj.getIsActive());
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        return entity;
    }

    private SopRouteBindingDO toDO(SopRouteBinding entity) {
        if (entity == null) {
            return null;
        }
        SopRouteBindingDO doObj = new SopRouteBindingDO();
        doObj.setId(entity.getId());
        doObj.setSopDocumentId(entity.getSopDocumentId());
        doObj.setRouteCode(entity.getRouteCode());
        doObj.setRouteName(entity.getRouteName());
        doObj.setStepNo(entity.getStepNo());
        doObj.setProcessCode(entity.getProcessCode());
        doObj.setProcessName(entity.getProcessName());
        doObj.setWorkstationId(entity.getWorkstationId());
        doObj.setWorkstationName(entity.getWorkstationName());
        doObj.setSortOrder(entity.getSortOrder());
        doObj.setIsActive(entity.getIsActive());
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        return doObj;
    }
}