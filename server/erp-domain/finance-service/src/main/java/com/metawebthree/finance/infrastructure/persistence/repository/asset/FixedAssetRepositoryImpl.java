package com.metawebthree.finance.infrastructure.persistence.repository.asset;

import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.metawebthree.finance.domain.repository.asset.FixedAssetRepository;
import com.metawebthree.finance.infrastructure.persistence.dataobject.asset.FixedAssetDO;
import com.metawebthree.finance.infrastructure.persistence.dataobject.asset.FixedAssetDepreciationDO;
import com.metawebthree.finance.infrastructure.persistence.dataobject.asset.FixedAssetDisposalDO;
import com.metawebthree.finance.infrastructure.persistence.dataobject.asset.FixedAssetInventoryDO;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * 固定资产仓储（本地开发用内存实现）。
 * 固定资产相关数据表尚未落地，这里以内存存储保证服务可独立启动与联调。
 */
@Repository
public class FixedAssetRepositoryImpl implements FixedAssetRepository {

    private final Map<Long, FixedAssetDO> assets = new ConcurrentHashMap<>();
    private final Map<Long, FixedAssetDepreciationDO> depreciations = new ConcurrentHashMap<>();
    private final Map<Long, FixedAssetInventoryDO> inventories = new ConcurrentHashMap<>();
    private final Map<Long, FixedAssetDisposalDO> disposals = new ConcurrentHashMap<>();

    @Override
    public FixedAssetDO save(FixedAssetDO asset) {
        if (asset.getId() == null) {
            asset.setId(IdWorker.getId());
        }
        assets.put(asset.getId(), asset);
        return asset;
    }

    @Override
    public void delete(Long id) {
        assets.remove(id);
    }

    @Override
    public FixedAssetDO findById(Long id) {
        return assets.get(id);
    }

    @Override
    public FixedAssetDO findByCode(String code) {
        return assets.values().stream()
                .filter(a -> Objects.equals(a.getAssetCode(), code))
                .findFirst()
                .orElse(null);
    }

    @Override
    public List<FixedAssetDO> findAll() {
        return assets.values().stream().collect(Collectors.toList());
    }

    @Override
    public List<FixedAssetDO> findByDepartment(Long departmentId) {
        return assets.values().stream()
                .filter(a -> Objects.equals(a.getDepartmentId(), departmentId))
                .collect(Collectors.toList());
    }

    @Override
    public List<FixedAssetDO> findByStatus(String status) {
        return assets.values().stream()
                .filter(a -> Objects.equals(a.getStatus(), status))
                .collect(Collectors.toList());
    }

    @Override
    public List<FixedAssetDO> findByCategory(String category) {
        return assets.values().stream()
                .filter(a -> Objects.equals(a.getAssetCategory(), category))
                .collect(Collectors.toList());
    }

    @Override
    public FixedAssetDepreciationDO saveDepreciation(FixedAssetDepreciationDO depreciation) {
        if (depreciation.getId() == null) {
            depreciation.setId(IdWorker.getId());
        }
        depreciations.put(depreciation.getId(), depreciation);
        return depreciation;
    }

    @Override
    public List<FixedAssetDepreciationDO> findDepreciationByAssetId(Long assetId) {
        return depreciations.values().stream()
                .filter(d -> Objects.equals(d.getAssetId(), assetId))
                .collect(Collectors.toList());
    }

    @Override
    public List<FixedAssetDepreciationDO> findDepreciationByPeriod(String period) {
        return depreciations.values().stream()
                .filter(d -> Objects.equals(d.getDepreciationPeriod(), period))
                .collect(Collectors.toList());
    }

    @Override
    public FixedAssetInventoryDO saveInventory(FixedAssetInventoryDO inventory) {
        if (inventory.getId() == null) {
            inventory.setId(IdWorker.getId());
        }
        inventories.put(inventory.getId(), inventory);
        return inventory;
    }

    @Override
    public List<FixedAssetInventoryDO> findInventoryByStatus(String status) {
        return inventories.values().stream()
                .filter(i -> Objects.equals(i.getStatus(), status))
                .collect(Collectors.toList());
    }

    @Override
    public List<FixedAssetInventoryDO> findInventoryByAssetId(Long assetId) {
        return inventories.values().stream()
                .filter(i -> Objects.equals(i.getId(), assetId) || Objects.equals(i.getAssetId(), assetId))
                .collect(Collectors.toList());
    }

    @Override
    public FixedAssetDisposalDO saveDisposal(FixedAssetDisposalDO disposal) {
        if (disposal.getId() == null) {
            disposal.setId(IdWorker.getId());
        }
        disposals.put(disposal.getId(), disposal);
        return disposal;
    }

    @Override
    public List<FixedAssetDisposalDO> findDisposalByStatus(String status) {
        return disposals.values().stream()
                .filter(d -> Objects.equals(d.getStatus(), status))
                .collect(Collectors.toList());
    }

    @Override
    public List<FixedAssetDisposalDO> findDisposalByAssetId(Long assetId) {
        return disposals.values().stream()
                .filter(d -> Objects.equals(d.getId(), assetId) || Objects.equals(d.getAssetId(), assetId))
                .collect(Collectors.toList());
    }
}
