package com.metawebthree.mes.infrastructure.persistence.repository;

import com.metawebthree.mes.domain.entity.MaterialRequirement;
import com.metawebthree.mes.domain.entity.MaterialRequirement.MaterialRequirementItem;
import com.metawebthree.mes.domain.repository.MaterialRequirementRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.MaterialRequirementDO;
import com.metawebthree.mes.infrastructure.persistence.dataobject.MaterialRequirementItemDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.MaterialRequirementMapper;
import com.metawebthree.mes.infrastructure.persistence.mapper.MaterialRequirementItemMapper;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class MaterialRequirementRepositoryImpl implements MaterialRequirementRepository {

    private final MaterialRequirementMapper mapper;
    private final MaterialRequirementItemMapper itemMapper;

    public MaterialRequirementRepositoryImpl(MaterialRequirementMapper mapper,
                                              MaterialRequirementItemMapper itemMapper) {
        this.mapper = mapper;
        this.itemMapper = itemMapper;
    }

    @Override
    public Optional<MaterialRequirement> findById(Long id) {
        MaterialRequirementDO obj = mapper.selectById(id);
        if (obj == null) return Optional.empty();
        
        MaterialRequirement entity = toEntity(obj);
        entity.setItems(loadItemsByRequirementId(id));
        return Optional.of(entity);
    }

    @Override
    public Optional<MaterialRequirement> findByRequirementNo(String requirementNo) {
        MaterialRequirementDO obj = mapper.findByRequirementNo(requirementNo);
        if (obj == null) return Optional.empty();
        
        MaterialRequirement entity = toEntity(obj);
        entity.setItems(loadItemsByRequirementId(obj.getId()));
        return Optional.of(entity);
    }

    @Override
    public Optional<MaterialRequirement> findByWorkOrderNo(String workOrderNo) {
        MaterialRequirementDO obj = mapper.findByWorkOrderNo(workOrderNo);
        if (obj == null) return Optional.empty();
        
        MaterialRequirement entity = toEntity(obj);
        entity.setItems(loadItemsByRequirementId(obj.getId()));
        return Optional.of(entity);
    }

    @Override
    public List<MaterialRequirement> findByStatus(String status) {
        List<MaterialRequirementDO> list = mapper.findByStatus(status);
        return list.stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<MaterialRequirement> findByWarehouseId(String warehouseId) {
        List<MaterialRequirementDO> list = mapper.findByWarehouseId(warehouseId);
        return list.stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<MaterialRequirement> findByWorkshopId(String workshopId) {
        List<MaterialRequirementDO> list = mapper.findByWorkshopId(workshopId);
        return list.stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<MaterialRequirement> findByProductCode(String productCode) {
        List<MaterialRequirementDO> list = mapper.findByProductCode(productCode);
        return list.stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public MaterialRequirement save(MaterialRequirement requirement) {
        MaterialRequirementDO DO = toDO(requirement);
        if (requirement.getId() == null) {
            mapper.insert(DO);
            requirement.setId(DO.getId());
        } else {
            mapper.updateById(DO);
        }
        
        // 保存子项
        if (requirement.getItems() != null && !requirement.getItems().isEmpty()) {
            for (MaterialRequirementItem item : requirement.getItems()) {
                item.setRequirementId(requirement.getId());
                saveRequirementItem(item);
            }
        }
        return requirement;
    }

    @Override
    public void update(MaterialRequirement requirement) {
        mapper.updateById(toDO(requirement));
    }

    @Override
    public void deleteById(Long id) {
        // 先删除子项
        itemMapper.deleteByRequirementId(id);
        mapper.deleteById(id);
    }

    private List<MaterialRequirementItem> loadItemsByRequirementId(Long requirementId) {
        List<MaterialRequirementItemDO> items = itemMapper.findByRequirementId(requirementId);
        return items.stream().map(this::toItemEntity).collect(Collectors.toList());
    }

    private void saveRequirementItem(MaterialRequirementItem item) {
        MaterialRequirementItemDO DO = toItemDO(item);
        if (item.getId() == null) {
            itemMapper.insert(DO);
            item.setId(DO.getId());
        } else {
            itemMapper.updateById(DO);
        }
    }

    // ========== Entity <-> DO 转换方法 ==========

    private MaterialRequirement toEntity(MaterialRequirementDO obj) {
        if (obj == null) return null;
        MaterialRequirement entity = new MaterialRequirement();
        entity.setId(obj.getId());
        entity.setRequirementNo(obj.getRequirementNo());
        entity.setWorkOrderNo(obj.getWorkOrderNo());
        entity.setProductCode(obj.getProductCode());
        entity.setProductName(obj.getProductName());
        entity.setQuantity(obj.getQuantity());
        entity.setBomVersion(obj.getBomVersion());
        entity.setStatus(obj.getStatus());
        entity.setWarehouseId(obj.getWarehouseId());
        entity.setWorkshopId(obj.getWorkshopId());
        entity.setRequirementType(obj.getRequirementType());
        entity.setRequiredDate(obj.getRequiredDate());
        entity.setCreatedBy(obj.getCreatedBy());
        entity.setUpdatedBy(obj.getUpdatedBy());
        entity.setCreatedAt(obj.getCreatedAt());
        entity.setUpdatedAt(obj.getUpdatedAt());
        return entity;
    }

    private MaterialRequirementDO toDO(MaterialRequirement entity) {
        if (entity == null) return null;
        MaterialRequirementDO DO = new MaterialRequirementDO();
        DO.setId(entity.getId());
        DO.setRequirementNo(entity.getRequirementNo());
        DO.setWorkOrderNo(entity.getWorkOrderNo());
        DO.setProductCode(entity.getProductCode());
        DO.setProductName(entity.getProductName());
        DO.setQuantity(entity.getQuantity());
        DO.setBomVersion(entity.getBomVersion());
        DO.setStatus(entity.getStatus());
        DO.setWarehouseId(entity.getWarehouseId());
        DO.setWorkshopId(entity.getWorkshopId());
        DO.setRequirementType(entity.getRequirementType());
        DO.setRequiredDate(entity.getRequiredDate());
        DO.setCreatedBy(entity.getCreatedBy());
        DO.setUpdatedBy(entity.getUpdatedBy());
        DO.setCreatedAt(entity.getCreatedAt());
        DO.setUpdatedAt(entity.getUpdatedAt());
        return DO;
    }

    private MaterialRequirementItem toItemEntity(MaterialRequirementItemDO obj) {
        if (obj == null) return null;
        MaterialRequirementItem entity = new MaterialRequirementItem();
        entity.setId(obj.getId());
        entity.setRequirementId(obj.getRequirementId());
        entity.setMaterialCode(obj.getMaterialCode());
        entity.setMaterialName(obj.getMaterialName());
        entity.setMaterialSpec(obj.getMaterialSpec());
        entity.setUnitCode(obj.getUnitCode());
        entity.setUnitName(obj.getUnitName());
        entity.setRequiredQuantity(obj.getRequiredQuantity());
        entity.setIssuedQuantity(obj.getIssuedQuantity());
        entity.setPendingQuantity(obj.getPendingQuantity());
        entity.setScrapQuantity(obj.getScrapQuantity());
        entity.setLocationId(obj.getLocationId());
        entity.setBatchNo(obj.getBatchNo());
        entity.setStatus(obj.getStatus());
        entity.setCreatedAt(obj.getCreatedAt());
        entity.setUpdatedAt(obj.getUpdatedAt());
        return entity;
    }

    private MaterialRequirementItemDO toItemDO(MaterialRequirementItem entity) {
        if (entity == null) return null;
        MaterialRequirementItemDO DO = new MaterialRequirementItemDO();
        DO.setId(entity.getId());
        DO.setRequirementId(entity.getRequirementId());
        DO.setMaterialCode(entity.getMaterialCode());
        DO.setMaterialName(entity.getMaterialName());
        DO.setMaterialSpec(entity.getMaterialSpec());
        DO.setUnitCode(entity.getUnitCode());
        DO.setUnitName(entity.getUnitName());
        DO.setRequiredQuantity(entity.getRequiredQuantity());
        DO.setIssuedQuantity(entity.getIssuedQuantity());
        DO.setPendingQuantity(entity.getPendingQuantity());
        DO.setScrapQuantity(entity.getScrapQuantity());
        DO.setLocationId(entity.getLocationId());
        DO.setBatchNo(entity.getBatchNo());
        DO.setStatus(entity.getStatus());
        DO.setCreatedAt(entity.getCreatedAt());
        DO.setUpdatedAt(entity.getUpdatedAt());
        return DO;
    }
}