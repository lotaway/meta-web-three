package com.metawebthree.mes.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.mes.domain.entity.SopDocument;
import com.metawebthree.mes.domain.entity.SopDocumentVersion;
import com.metawebthree.mes.domain.repository.SopDocumentRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.SopDocumentDO;
import com.metawebthree.mes.infrastructure.persistence.dataobject.SopDocumentVersionDO;
import com.metawebthree.mes.infrastructure.persistence.dataobject.SopRouteBindingDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.SopDocumentMapper;
import com.metawebthree.mes.infrastructure.persistence.mapper.SopDocumentVersionMapper;
import com.metawebthree.mes.infrastructure.persistence.mapper.SopRouteBindingMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class SopDocumentRepositoryImpl implements SopDocumentRepository {

    @Autowired
    private SopDocumentMapper sopDocumentMapper;

    @Autowired
    private SopDocumentVersionMapper sopDocumentVersionMapper;

    @Autowired
    private SopRouteBindingMapper sopRouteBindingMapper;

    @Override
    public SopDocument save(SopDocument sopDocument) {
        SopDocumentDO doObj = toDO(sopDocument);
        if (sopDocument.getId() == null) {
            sopDocumentMapper.insert(doObj);
            sopDocument.setId(doObj.getId());
        } else {
            sopDocumentMapper.updateById(doObj);
        }
        return sopDocument;
    }

    @Override
    public Optional<SopDocument> findById(Long id) {
        SopDocumentDO docDO = sopDocumentMapper.selectById(id);
        if (docDO == null) {
            return Optional.empty();
        }
        SopDocument entity = toEntity(docDO);
        loadRelations(entity);
        return Optional.of(entity);
    }

    @Override
    public Optional<SopDocument> findByDocumentCode(String documentCode) {
        LambdaQueryWrapper<SopDocumentDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(SopDocumentDO::getDocumentCode, documentCode);
        SopDocumentDO docDO = sopDocumentMapper.selectOne(wrapper);
        if (docDO == null) {
            return Optional.empty();
        }
        SopDocument entity = toEntity(docDO);
        loadRelations(entity);
        return Optional.of(entity);
    }

    @Override
    public List<SopDocument> findAll() {
        List<SopDocumentDO> doList = sopDocumentMapper.selectList(null);
        return doList.stream().map(doObj -> {
            SopDocument entity = toEntity(doObj);
            loadRelations(entity);
            return entity;
        }).collect(Collectors.toList());
    }

    @Override
    public List<SopDocument> findByStatus(SopDocument.SopStatus status) {
        LambdaQueryWrapper<SopDocumentDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(SopDocumentDO::getStatus, status.name());
        List<SopDocumentDO> doList = sopDocumentMapper.selectList(wrapper);
        return doList.stream().map(doObj -> {
            SopDocument entity = toEntity(doObj);
            loadRelations(entity);
            return entity;
        }).collect(Collectors.toList());
    }

    @Override
    public List<SopDocument> findByDocumentType(String documentType) {
        LambdaQueryWrapper<SopDocumentDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(SopDocumentDO::getDocumentType, documentType);
        List<SopDocumentDO> doList = sopDocumentMapper.selectList(wrapper);
        return doList.stream().map(doObj -> {
            SopDocument entity = toEntity(doObj);
            loadRelations(entity);
            return entity;
        }).collect(Collectors.toList());
    }

    @Override
    public List<SopDocument> findByCategory(String category) {
        LambdaQueryWrapper<SopDocumentDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(SopDocumentDO::getCategory, category);
        List<SopDocumentDO> doList = sopDocumentMapper.selectList(wrapper);
        return doList.stream().map(doObj -> {
            SopDocument entity = toEntity(doObj);
            loadRelations(entity);
            return entity;
        }).collect(Collectors.toList());
    }

    @Override
    public boolean existsByDocumentCode(String documentCode) {
        LambdaQueryWrapper<SopDocumentDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(SopDocumentDO::getDocumentCode, documentCode);
        return sopDocumentMapper.selectCount(wrapper) > 0;
    }

    @Override
    public void deleteById(Long id) {
        sopDocumentMapper.deleteById(id);
    }

    @Override
    public List<SopDocument> findByDocumentNameContaining(String documentName) {
        LambdaQueryWrapper<SopDocumentDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.like(SopDocumentDO::getDocumentName, documentName);
        List<SopDocumentDO> doList = sopDocumentMapper.selectList(wrapper);
        return doList.stream().map(doObj -> {
            SopDocument entity = toEntity(doObj);
            loadRelations(entity);
            return entity;
        }).collect(Collectors.toList());
    }

    private void loadRelations(SopDocument entity) {
        if (entity.getId() == null) {
            return;
        }
        LambdaQueryWrapper<SopDocumentVersionDO> versionWrapper = new LambdaQueryWrapper<>();
        versionWrapper.eq(SopDocumentVersionDO::getSopDocumentId, entity.getId());
        List<SopDocumentVersionDO> versionDOs = sopDocumentVersionMapper.selectList(versionWrapper);
        if (versionDOs != null && !versionDOs.isEmpty()) {
            entity.setVersions(versionDOs.stream().map(this::versionToEntity).collect(Collectors.toList()));
        }

        LambdaQueryWrapper<SopRouteBindingDO> bindingWrapper = new LambdaQueryWrapper<>();
        bindingWrapper.eq(SopRouteBindingDO::getSopDocumentId, entity.getId());
        List<SopRouteBindingDO> bindingDOs = sopRouteBindingMapper.selectList(bindingWrapper);
        if (bindingDOs != null && !bindingDOs.isEmpty()) {
            entity.setRouteBindings(bindingDOs.stream().map(this::bindingToEntity).collect(Collectors.toList()));
        }
    }

    private SopDocument toEntity(SopDocumentDO doObj) {
        if (doObj == null) {
            return null;
        }
        SopDocument entity = new SopDocument();
        entity.setId(doObj.getId());
        entity.setDocumentCode(doObj.getDocumentCode());
        entity.setDocumentName(doObj.getDocumentName());
        entity.setDocumentType(doObj.getDocumentType());
        entity.setCategory(doObj.getCategory());
        entity.setCurrentVersion(doObj.getCurrentVersion());
        entity.setStatus(SopDocument.SopStatus.valueOf(doObj.getStatus()));
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        return entity;
    }

    private SopDocumentDO toDO(SopDocument entity) {
        if (entity == null) {
            return null;
        }
        SopDocumentDO doObj = new SopDocumentDO();
        doObj.setId(entity.getId());
        doObj.setDocumentCode(entity.getDocumentCode());
        doObj.setDocumentName(entity.getDocumentName());
        doObj.setDocumentType(entity.getDocumentType());
        doObj.setCategory(entity.getCategory());
        doObj.setCurrentVersion(entity.getCurrentVersionNo());
        doObj.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        return doObj;
    }

    private SopDocumentVersion versionToEntity(SopDocumentVersionDO doObj) {
        if (doObj == null) {
            return null;
        }
        SopDocumentVersion entity = new SopDocumentVersion();
        entity.setId(doObj.getId());
        entity.setSopDocumentId(doObj.getSopDocumentId());
        entity.setVersionNo(doObj.getVersionNo());
        entity.setFileName(doObj.getFileName());
        entity.setFilePath(doObj.getFilePath());
        entity.setFileType(doObj.getFileType());
        entity.setFileSize(doObj.getFileSize());
        entity.setUploader(doObj.getUploader());
        entity.setChangeDescription(doObj.getChangeDescription());
        entity.setIsCurrentVersion(doObj.getIsCurrentVersion());
        entity.setUploadedAt(doObj.getUploadedAt());
        entity.setCreatedAt(doObj.getCreatedAt());
        return entity;
    }

    private com.metawebthree.mes.domain.entity.SopRouteBinding bindingToEntity(SopRouteBindingDO doObj) {
        if (doObj == null) {
            return null;
        }
        com.metawebthree.mes.domain.entity.SopRouteBinding entity = new com.metawebthree.mes.domain.entity.SopRouteBinding();
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
}