package com.metawebthree.mes.infrastructure.persistence.repository;

import com.metawebthree.mes.domain.entity.BomBillOfMaterials;
import com.metawebthree.mes.domain.entity.BomItem;
import com.metawebthree.mes.domain.entity.BomVersion;
import com.metawebthree.mes.domain.entity.ProcessBomItem;
import com.metawebthree.mes.domain.repository.BomRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.BomBillOfMaterialsDO;
import com.metawebthree.mes.infrastructure.persistence.dataobject.BomItemDO;
import com.metawebthree.mes.infrastructure.persistence.dataobject.BomVersionDO;
import com.metawebthree.mes.infrastructure.persistence.dataobject.ProcessBomItemDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.BomBillOfMaterialsMapper;
import com.metawebthree.mes.infrastructure.persistence.mapper.BomItemMapper;
import com.metawebthree.mes.infrastructure.persistence.mapper.BomVersionMapper;
import com.metawebthree.mes.infrastructure.persistence.mapper.ProcessBomItemMapper;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class BomRepositoryImpl implements BomRepository {

    private final BomBillOfMaterialsMapper bomMapper;
    private final BomItemMapper itemMapper;
    private final BomVersionMapper versionMapper;
    private final ProcessBomItemMapper processBomMapper;

    public BomRepositoryImpl(BomBillOfMaterialsMapper bomMapper,
                             BomItemMapper itemMapper,
                             BomVersionMapper versionMapper,
                             ProcessBomItemMapper processBomMapper) {
        this.bomMapper = bomMapper;
        this.itemMapper = itemMapper;
        this.versionMapper = versionMapper;
        this.processBomMapper = processBomMapper;
    }

    @Override
    public Optional<BomBillOfMaterials> findById(Long id) {
        BomBillOfMaterialsDO obj = bomMapper.selectById(id);
        if (obj == null) return Optional.empty();
        
        BomBillOfMaterials entity = toEntity(obj);
        entity.setItems(loadItemsByBomId(id));
        return Optional.of(entity);
    }

    @Override
    public Optional<BomBillOfMaterials> findByBomCode(String bomCode) {
        BomBillOfMaterialsDO obj = bomMapper.findByBomCode(bomCode);
        if (obj == null) return Optional.empty();
        
        BomBillOfMaterials entity = toEntity(obj);
        entity.setItems(loadItemsByBomId(obj.getId()));
        return Optional.of(entity);
    }

    @Override
    public List<BomBillOfMaterials> findByProductCode(String productCode) {
        List<BomBillOfMaterialsDO> list = bomMapper.findByProductCode(productCode);
        return list.stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<BomBillOfMaterials> findActiveByProductCode(String productCode) {
        List<BomBillOfMaterialsDO> list = bomMapper.findActiveByProductCode(productCode);
        return list.stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<BomBillOfMaterials> findByProductCodeAndVersion(String productCode, String version) {
        List<BomBillOfMaterialsDO> list = bomMapper.findByProductCodeAndVersion(productCode, version);
        return list.stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public BomBillOfMaterials save(BomBillOfMaterials bom) {
        BomBillOfMaterialsDO DO = toDO(bom);
        if (bom.getId() == null) {
            bomMapper.insert(DO);
            bom.setId(DO.getId());
        } else {
            bomMapper.updateById(DO);
        }
        
        // 保存子项
        if (bom.getItems() != null && !bom.getItems().isEmpty()) {
            for (BomItem item : bom.getItems()) {
                item.setBomId(bom.getId());
                saveBomItem(item);
            }
        }
        return bom;
    }

    @Override
    public void update(BomBillOfMaterials bom) {
        bomMapper.updateById(toDO(bom));
    }

    @Override
    public void deleteById(Long id) {
        // 先删除子项
        itemMapper.deleteByBomId(id);
        bomMapper.deleteById(id);
    }

    @Override
    public Optional<BomVersion> findVersionByProductCode(String productCode) {
        BomVersionDO obj = versionMapper.findByProductCode(productCode);
        if (obj == null) return Optional.empty();
        return Optional.of(toVersionEntity(obj));
    }

    @Override
    public BomVersion saveVersion(BomVersion version) {
        BomVersionDO DO = toVersionDO(version);
        if (version.getId() == null) {
            versionMapper.insert(DO);
            version.setId(DO.getId());
        } else {
            versionMapper.updateById(DO);
        }
        return version;
    }

    @Override
    public List<ProcessBomItem> findProcessBomByRouteId(String processRouteId) {
        List<ProcessBomItemDO> list = processBomMapper.findByRouteId(processRouteId);
        return list.stream().map(this::toProcessBomEntity).collect(Collectors.toList());
    }

    @Override
    public List<ProcessBomItem> findProcessBomByProcessCode(String processRouteId, String processCode) {
        List<ProcessBomItemDO> list = processBomMapper.findByProcessCode(processRouteId, processCode);
        return list.stream().map(this::toProcessBomEntity).collect(Collectors.toList());
    }

    private List<BomItem> loadItemsByBomId(Long bomId) {
        List<BomItemDO> items = itemMapper.findByBomId(bomId);
        return items.stream().map(this::toItemEntity).collect(Collectors.toList());
    }

    private void saveBomItem(BomItem item) {
        BomItemDO DO = toItemDO(item);
        if (item.getId() == null) {
            itemMapper.insert(DO);
            item.setId(DO.getId());
        } else {
            itemMapper.updateById(DO);
        }
    }

    // ========== Entity <-> DO 转换方法 ==========

    private BomBillOfMaterials toEntity(BomBillOfMaterialsDO obj) {
        if (obj == null) return null;
        BomBillOfMaterials entity = new BomBillOfMaterials();
        entity.setId(obj.getId());
        entity.setBomCode(obj.getBomCode());
        entity.setProductCode(obj.getProductCode());
        entity.setProductName(obj.getProductName());
        entity.setVersion(obj.getVersion());
        entity.setVersionStatus(obj.getVersionStatus());
        entity.setEffectiveDate(obj.getEffectiveDate());
        entity.setExpiryDate(obj.getExpiryDate());
        entity.setBomType(obj.getBomType());
        entity.setProcessRouteId(obj.getProcessRouteId());
        entity.setDescription(obj.getDescription());
        entity.setStatus(obj.getStatus());
        entity.setItemCount(obj.getItemCount());
        entity.setPreviousVersion(obj.getPreviousVersion());
        entity.setChangeReason(obj.getChangeReason());
        entity.setCreatedBy(obj.getCreatedBy());
        entity.setUpdatedBy(obj.getUpdatedBy());
        entity.setCreatedAt(obj.getCreatedAt());
        entity.setUpdatedAt(obj.getUpdatedAt());
        return entity;
    }

    private BomBillOfMaterialsDO toDO(BomBillOfMaterials entity) {
        if (entity == null) return null;
        BomBillOfMaterialsDO DO = new BomBillOfMaterialsDO();
        DO.setId(entity.getId());
        DO.setBomCode(entity.getBomCode());
        DO.setProductCode(entity.getProductCode());
        DO.setProductName(entity.getProductName());
        DO.setVersion(entity.getVersion());
        DO.setVersionStatus(entity.getVersionStatus());
        DO.setEffectiveDate(entity.getEffectiveDate());
        DO.setExpiryDate(entity.getExpiryDate());
        DO.setBomType(entity.getBomType());
        DO.setProcessRouteId(entity.getProcessRouteId());
        DO.setDescription(entity.getDescription());
        DO.setStatus(entity.getStatus());
        DO.setItemCount(entity.getItemCount());
        DO.setPreviousVersion(entity.getPreviousVersion());
        DO.setChangeReason(entity.getChangeReason());
        DO.setCreatedBy(entity.getCreatedBy());
        DO.setUpdatedBy(entity.getUpdatedBy());
        DO.setCreatedAt(entity.getCreatedAt());
        DO.setUpdatedAt(entity.getUpdatedAt());
        return DO;
    }

    private BomItem toItemEntity(BomItemDO obj) {
        if (obj == null) return null;
        BomItem entity = new BomItem();
        entity.setId(obj.getId());
        entity.setBomId(obj.getBomId());
        entity.setMaterialCode(obj.getMaterialCode());
        entity.setMaterialName(obj.getMaterialName());
        entity.setMaterialSpec(obj.getMaterialSpec());
        entity.setUnitCode(obj.getUnitCode());
        entity.setUnitName(obj.getUnitName());
        entity.setQuantity(obj.getQuantity());
        entity.setScrapRate(obj.getScrapRate());
        entity.setSequence(obj.getSequence());
        entity.setLevel(obj.getLevel());
        entity.setParentMaterialCode(obj.getParentMaterialCode());
        entity.setItemType(obj.getItemType());
        entity.setPosition(obj.getPosition());
        entity.setRemark(obj.getRemark());
        entity.setStatus(obj.getStatus());
        entity.setSubstituteItemId(obj.getSubstituteItemId());
        entity.setCreatedBy(obj.getCreatedBy());
        entity.setUpdatedBy(obj.getUpdatedBy());
        entity.setCreatedAt(obj.getCreatedAt());
        entity.setUpdatedAt(obj.getUpdatedAt());
        return entity;
    }

    private BomItemDO toItemDO(BomItem entity) {
        if (entity == null) return null;
        BomItemDO DO = new BomItemDO();
        DO.setId(entity.getId());
        DO.setBomId(entity.getBomId());
        DO.setMaterialCode(entity.getMaterialCode());
        DO.setMaterialName(entity.getMaterialName());
        DO.setMaterialSpec(entity.getMaterialSpec());
        DO.setUnitCode(entity.getUnitCode());
        DO.setUnitName(entity.getUnitName());
        DO.setQuantity(entity.getQuantity());
        DO.setScrapRate(entity.getScrapRate());
        DO.setSequence(entity.getSequence());
        DO.setLevel(entity.getLevel());
        DO.setParentMaterialCode(entity.getParentMaterialCode());
        DO.setItemType(entity.getItemType());
        DO.setPosition(entity.getPosition());
        DO.setRemark(entity.getRemark());
        DO.setStatus(entity.getStatus());
        DO.setSubstituteItemId(entity.getSubstituteItemId());
        DO.setCreatedBy(entity.getCreatedBy());
        DO.setUpdatedBy(entity.getUpdatedBy());
        DO.setCreatedAt(entity.getCreatedAt());
        DO.setUpdatedAt(entity.getUpdatedAt());
        return DO;
    }

    private BomVersion toVersionEntity(BomVersionDO obj) {
        if (obj == null) return null;
        BomVersion entity = new BomVersion();
        entity.setId(obj.getId());
        entity.setProductCode(obj.getProductCode());
        entity.setProductName(obj.getProductName());
        entity.setCreatedAt(obj.getCreatedAt());
        entity.setUpdatedAt(obj.getUpdatedAt());
        return entity;
    }

    private BomVersionDO toVersionDO(BomVersion entity) {
        if (entity == null) return null;
        BomVersionDO DO = new BomVersionDO();
        DO.setId(entity.getId());
        DO.setProductCode(entity.getProductCode());
        DO.setProductName(entity.getProductName());
        DO.setCreatedAt(entity.getCreatedAt());
        DO.setUpdatedAt(entity.getUpdatedAt());
        return DO;
    }

    private ProcessBomItem toProcessBomEntity(ProcessBomItemDO obj) {
        if (obj == null) return null;
        ProcessBomItem entity = new ProcessBomItem();
        entity.setId(obj.getId());
        entity.setProcessBomCode(obj.getProcessBomCode());
        entity.setProductCode(obj.getProductCode());
        entity.setProcessRouteId(obj.getProcessRouteId());
        entity.setProcessCode(obj.getProcessCode());
        entity.setProcessName(obj.getProcessName());
        entity.setVersion(obj.getVersion());
        entity.setStatus(obj.getStatus());
        entity.setDescription(obj.getDescription());
        entity.setCreatedBy(obj.getCreatedBy());
        entity.setUpdatedBy(obj.getUpdatedBy());
        entity.setCreatedAt(obj.getCreatedAt());
        entity.setUpdatedAt(obj.getUpdatedAt());
        return entity;
    }
}