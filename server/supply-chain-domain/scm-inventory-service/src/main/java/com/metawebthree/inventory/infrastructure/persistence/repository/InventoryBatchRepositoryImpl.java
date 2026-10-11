package com.metawebthree.inventory.infrastructure.persistence.repository;

import com.metawebthree.inventory.domain.entity.InventoryBatch;
import com.metawebthree.inventory.domain.repository.InventoryBatchRepository;
import com.metawebthree.inventory.infrastructure.persistence.converter.InventoryBatchConverter;
import com.metawebthree.inventory.infrastructure.persistence.dataobject.InventoryBatchDO;
import com.metawebthree.inventory.infrastructure.persistence.mapper.InventoryBatchMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;

@Repository
public class InventoryBatchRepositoryImpl implements InventoryBatchRepository {

    @Autowired
    private InventoryBatchMapper mapper;

    @Autowired
    private InventoryBatchConverter converter;

    @Override
    public InventoryBatch findById(Long id) {
        InventoryBatchDO dataObject = mapper.selectById(id);
        return converter.toEntity(dataObject);
    }

    @Override
    public InventoryBatch findByBatchNo(String skuCode, Long warehouseId, String batchNo) {
        List<InventoryBatchDO> list = mapper.selectByBatchNo(skuCode, warehouseId, batchNo);
        if (list == null || list.isEmpty()) {
            return null;
        }
        return converter.toEntity(list.get(0));
    }

    @Override
    public List<InventoryBatch> findAvailableBatches(String skuCode, Long warehouseId) {
        List<InventoryBatchDO> list = mapper.selectAvailableBatchesFifo(skuCode, warehouseId);
        return converter.toEntityList(list);
    }

    @Override
    public List<InventoryBatch> findByWarehouseAndSku(Long warehouseId, String skuCode) {
        return converter.toEntityList(
                mapper.selectList(new com.baomidou.mybatisplus.core.conditions.query.QueryWrapper<InventoryBatchDO>()
                        .eq("sku_code", skuCode)
                        .eq("warehouse_id", warehouseId)
                        .orderByAsc("inbound_date")
                        .orderByAsc("id"))
        );
    }

    @Override
    public InventoryBatch save(InventoryBatch batch) {
        batch.setCreatedAt(LocalDateTime.now());
        batch.setUpdatedAt(LocalDateTime.now());
        if (batch.getVersion() == null) {
            batch.setVersion(0);
        }
        mapper.insert(converter.toDataObject(batch));
        return batch;
    }

    @Override
    public boolean update(InventoryBatch batch) {
        batch.setUpdatedAt(LocalDateTime.now());
        return mapper.updateById(converter.toDataObject(batch)) > 0;
    }

    @Override
    public boolean delete(Long id) {
        return mapper.deleteById(id) > 0;
    }
}