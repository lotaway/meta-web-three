package com.metawebthree.finance.domain.repository.asset;

import com.metawebthree.finance.infrastructure.persistence.dataobject.asset.FixedAssetDO;
import com.metawebthree.finance.infrastructure.persistence.dataobject.asset.FixedAssetDepreciationDO;
import com.metawebthree.finance.infrastructure.persistence.dataobject.asset.FixedAssetInventoryDO;
import com.metawebthree.finance.infrastructure.persistence.dataobject.asset.FixedAssetDisposalDO;

import java.util.List;

public interface FixedAssetRepository {
    FixedAssetDO save(FixedAssetDO asset);
    void delete(Long id);
    FixedAssetDO findById(Long id);
    FixedAssetDO findByCode(String code);
    List<FixedAssetDO> findAll();
    List<FixedAssetDO> findByDepartment(Long departmentId);
    List<FixedAssetDO> findByStatus(String status);
    List<FixedAssetDO> findByCategory(String category);

    FixedAssetDepreciationDO saveDepreciation(FixedAssetDepreciationDO depreciation);
    List<FixedAssetDepreciationDO> findDepreciationByAssetId(Long assetId);
    List<FixedAssetDepreciationDO> findDepreciationByPeriod(String period);

    FixedAssetInventoryDO saveInventory(FixedAssetInventoryDO inventory);
    List<FixedAssetInventoryDO> findInventoryByStatus(String status);
    List<FixedAssetInventoryDO> findInventoryByAssetId(Long assetId);

    FixedAssetDisposalDO saveDisposal(FixedAssetDisposalDO disposal);
    List<FixedAssetDisposalDO> findDisposalByStatus(String status);
    List<FixedAssetDisposalDO> findDisposalByAssetId(Long assetId);
}