package com.metawebthree.finance.application.command.asset;

import com.metawebthree.finance.application.command.asset.dto.AssetDisposalCreateCommand;
import com.metawebthree.finance.application.command.asset.dto.AssetInventoryCreateCommand;
import com.metawebthree.finance.application.command.asset.dto.DepreciationGenerateCommand;
import com.metawebthree.finance.application.command.asset.dto.FixedAssetCreateCommand;
import com.metawebthree.finance.domain.repository.asset.FixedAssetRepository;
import com.metawebthree.finance.infrastructure.persistence.dataobject.asset.FixedAssetDO;
import com.metawebthree.finance.infrastructure.persistence.dataobject.asset.FixedAssetDepreciationDO;
import com.metawebthree.finance.infrastructure.persistence.dataobject.asset.FixedAssetDisposalDO;
import com.metawebthree.finance.infrastructure.persistence.dataobject.asset.FixedAssetInventoryDO;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.LocalDateTime;
import java.time.YearMonth;
import java.util.List;

@Service
public class FixedAssetCommandService {
    private final FixedAssetRepository repository;

    public FixedAssetCommandService(FixedAssetRepository repository) {
        this.repository = repository;
    }

    public Long createAsset(FixedAssetCreateCommand command) {
        FixedAssetDO asset = new FixedAssetDO();
        asset.setAssetCode(command.getAssetCode());
        asset.setAssetName(command.getAssetName());
        asset.setAssetCategory(command.getAssetCategory());
        asset.setSpecification(command.getSpecification());
        asset.setModel(command.getModel());
        asset.setSerialNumber(command.getSerialNumber());
        asset.setSupplierId(command.getSupplierId());
        asset.setSupplierName(command.getSupplierName());
        asset.setManufacturer(command.getManufacturer());
        asset.setPurchaseDate(command.getPurchaseDate());
        asset.setOriginalValue(command.getOriginalValue());
        asset.setResidualValue(command.getResidualValue() != null ? command.getResidualValue() : BigDecimal.ZERO);
        asset.setUsefulLife(command.getUsefulLife());
        asset.setDepreciationMethod(command.getDepreciationMethod());
        asset.setDepartmentId(command.getDepartmentId());
        asset.setDepartmentName(command.getDepartmentName());
        asset.setLocation(command.getLocation());
        asset.setCustodian(command.getCustodian());
        asset.setStatus("ACTIVE");
        asset.setUsageStatus("IN_USE");
        asset.setRemark(command.getRemark());
        asset.setCreatedBy(command.getCreatedBy());
        asset.setCreatorName(command.getCreatorName());
        asset.setCreatedAt(LocalDateTime.now());
        asset.setUpdatedAt(LocalDateTime.now());

        calculateDepreciation(asset);
        repository.save(asset);
        return asset.getId();
    }

    public void updateAsset(FixedAssetCreateCommand command) {
        FixedAssetDO asset = repository.findById(command.getId());
        if (asset != null) {
            asset.setAssetName(command.getAssetName());
            asset.setAssetCategory(command.getAssetCategory());
            asset.setSpecification(command.getSpecification());
            asset.setModel(command.getModel());
            asset.setDepartmentId(command.getDepartmentId());
            asset.setDepartmentName(command.getDepartmentName());
            asset.setLocation(command.getLocation());
            asset.setCustodian(command.getCustodian());
            asset.setRemark(command.getRemark());
            asset.setUpdatedAt(LocalDateTime.now());
            repository.save(asset);
        }
    }

    public void deleteAsset(Long id) {
        repository.delete(id);
    }

    public void transferAsset(Long assetId, Long newDepartmentId, String newDepartmentName, String newLocation, String newCustodian) {
        FixedAssetDO asset = repository.findById(assetId);
        if (asset != null) {
            asset.setDepartmentId(newDepartmentId);
            asset.setDepartmentName(newDepartmentName);
            asset.setLocation(newLocation);
            asset.setCustodian(newCustodian);
            asset.setUpdatedAt(LocalDateTime.now());
            repository.save(asset);
        }
    }

    public void generateDepreciation(DepreciationGenerateCommand command) {
        List<FixedAssetDO> assets;
        if (command.getDepartmentId() != null) {
            assets = repository.findByDepartment(command.getDepartmentId());
        } else {
            assets = repository.findAll();
        }

        String method = command.getDepreciationMethod();
        String period = command.getDepreciationPeriod();

        for (FixedAssetDO asset : assets) {
            if (!"ACTIVE".equals(asset.getStatus())) {
                continue;
            }

            FixedAssetDepreciationDO depreciation = new FixedAssetDepreciationDO();
            depreciation.setAssetId(asset.getId());
            depreciation.setAssetCode(asset.getAssetCode());
            depreciation.setAssetName(asset.getAssetName());
            depreciation.setDepreciationPeriod(period);
            depreciation.setDepreciationMethod(asset.getDepreciationMethod());
            depreciation.setOriginalValue(asset.getOriginalValue());
            depreciation.setResidualValue(asset.getResidualValue());
            depreciation.setUsefulLife(asset.getUsefulLife());
            depreciation.setDepreciationDate(period);
            depreciation.setStatus("PENDING");

            BigDecimal depreciationAmount = calculateDepreciationAmount(asset, method);
            depreciation.setDepreciationAmount(depreciationAmount);

            BigDecimal newAccumulated = asset.getAccumulatedDepreciation() != null 
                ? asset.getAccumulatedDepreciation().add(depreciationAmount)
                : depreciationAmount;
            depreciation.setAccumulatedDepreciation(newAccumulated);

            BigDecimal netValue = asset.getOriginalValue().subtract(newAccumulated);
            depreciation.setNetBookValue(netValue);

            depreciation.setCreatedAt(LocalDateTime.now());
            depreciation.setUpdatedAt(LocalDateTime.now());

            repository.saveDepreciation(depreciation);

            asset.setAccumulatedDepreciation(newAccumulated);
            asset.setNetValue(netValue);
            asset.setMonthlyDepreciation(depreciationAmount);
            asset.setUpdatedAt(LocalDateTime.now());
            repository.save(asset);
        }
    }

    private void calculateDepreciation(FixedAssetDO asset) {
        String method = asset.getDepreciationMethod();
        BigDecimal originalValue = asset.getOriginalValue();
        BigDecimal residualValue = asset.getResidualValue();
        Integer usefulLife = asset.getUsefulLife();

        if (originalValue == null || usefulLife == null || usefulLife == 0) {
            asset.setNetValue(originalValue);
            asset.setAccumulatedDepreciation(BigDecimal.ZERO);
            asset.setMonthlyDepreciation(BigDecimal.ZERO);
            return;
        }

        BigDecimal depreciableAmount = originalValue.subtract(residualValue != null ? residualValue : BigDecimal.ZERO);
        BigDecimal monthlyDepreciation;

        switch (method) {
            case "STRAIGHT_LINE":
                monthlyDepreciation = depreciableAmount.divide(BigDecimal.valueOf(usefulLife * 12), 2, RoundingMode.HALF_UP);
                asset.setAnnualDepreciationRate(BigDecimal.valueOf(100.0 / usefulLife));
                break;
            case "DOUBLE_DECLINING":
                double rate = 2.0 / usefulLife;
                asset.setAnnualDepreciationRate(BigDecimal.valueOf(rate * 100));
                double monthlyRate = rate / 12;
                monthlyDepreciation = originalValue.multiply(BigDecimal.valueOf(monthlyRate)).setScale(2, RoundingMode.HALF_UP);
                break;
            case "SUM_OF_YEARS":
                int sumOfYears = usefulLife * (usefulLife + 1) / 2;
                int currentYear = 1;
                double yearDepreciation = (double) (usefulLife - currentYear + 1) / sumOfYears;
                monthlyDepreciation = depreciableAmount.multiply(BigDecimal.valueOf(yearDepreciation))
                    .divide(BigDecimal.valueOf(12), 2, RoundingMode.HALF_UP);
                asset.setAnnualDepreciationRate(BigDecimal.valueOf(yearDepreciation * 100));
                break;
            default:
                monthlyDepreciation = depreciableAmount.divide(BigDecimal.valueOf(usefulLife * 12), 2, RoundingMode.HALF_UP);
                asset.setAnnualDepreciationRate(BigDecimal.valueOf(100.0 / usefulLife));
        }

        asset.setMonthlyDepreciation(monthlyDepreciation);
        asset.setAccumulatedDepreciation(BigDecimal.ZERO);
        asset.setNetValue(originalValue.subtract(residualValue != null ? residualValue : BigDecimal.ZERO));
    }

    private BigDecimal calculateDepreciationAmount(FixedAssetDO asset, String method) {
        BigDecimal originalValue = asset.getOriginalValue();
        BigDecimal residualValue = asset.getResidualValue() != null ? asset.getResidualValue() : BigDecimal.ZERO;
        Integer usefulLife = asset.getUsefulLife();

        if (originalValue == null || usefulLife == null || usefulLife == 0) {
            return BigDecimal.ZERO;
        }

        BigDecimal depreciableAmount = originalValue.subtract(residualValue);

        switch (method != null ? method : asset.getDepreciationMethod()) {
            case "STRAIGHT_LINE":
                return depreciableAmount.divide(BigDecimal.valueOf(usefulLife * 12), 2, RoundingMode.HALF_UP);
            case "DOUBLE_DECLINING":
                BigDecimal accumulated = asset.getAccumulatedDepreciation() != null ? asset.getAccumulatedDepreciation() : BigDecimal.ZERO;
                BigDecimal bookValue = originalValue.subtract(accumulated);
                double rate = 2.0 / usefulLife;
                return bookValue.multiply(BigDecimal.valueOf(rate / 12)).setScale(2, RoundingMode.HALF_UP);
            case "SUM_OF_YEARS":
                int sumOfYears = usefulLife * (usefulLife + 1) / 2;
                String currentPeriod = YearMonth.now().toString();
                int year = YearMonth.parse(currentPeriod).getYear();
                int remainingYears = usefulLife - ((2026 - Integer.parseInt(asset.getPurchaseDate().substring(0, 4))) - 1);
                remainingYears = Math.max(1, Math.min(remainingYears, usefulLife));
                double yearRatio = (double) remainingYears / sumOfYears;
                return depreciableAmount.multiply(BigDecimal.valueOf(yearRatio))
                    .divide(BigDecimal.valueOf(12), 2, RoundingMode.HALF_UP);
            default:
                return depreciableAmount.divide(BigDecimal.valueOf(usefulLife * 12), 2, RoundingMode.HALF_UP);
        }
    }

    public Long createInventory(AssetInventoryCreateCommand command) {
        FixedAssetInventoryDO inventory = new FixedAssetInventoryDO();
        inventory.setInventoryCode(command.getInventoryCode());
        inventory.setInventoryName(command.getInventoryName());
        inventory.setInventoryDate(command.getInventoryDate());
        inventory.setDepartmentId(command.getDepartmentId());
        inventory.setDepartmentName(command.getDepartmentName());
        inventory.setInventoryPerson(command.getInventoryPerson());
        inventory.setAssetId(command.getAssetId());
        inventory.setAssetCode(command.getAssetCode());
        inventory.setBookLocation(command.getBookLocation());
        inventory.setActualLocation(command.getActualLocation());
        inventory.setInventoryResult(command.getInventoryResult());
        inventory.setDiscrepancyReason(command.getDiscrepancyReason());
        inventory.setHandleMethod(command.getHandleMethod());
        inventory.setStatus("PENDING");
        inventory.setRemark(command.getRemark());
        inventory.setCreatedBy(command.getCreatedBy());
        inventory.setCreatorName(command.getCreatorName());
        inventory.setCreatedAt(LocalDateTime.now());
        inventory.setUpdatedAt(LocalDateTime.now());
        repository.saveInventory(inventory);
        return inventory.getId();
    }

    public void confirmInventory(Long inventoryId, String handleResult) {
        FixedAssetInventoryDO inventory = repository.findInventoryByAssetId(inventoryId).stream().findFirst().orElse(null);
        if (inventory != null) {
            inventory.setHandleResult(handleResult);
            inventory.setStatus("COMPLETED");
            inventory.setUpdatedAt(LocalDateTime.now());
            repository.saveInventory(inventory);
        }
    }

    public Long createDisposal(AssetDisposalCreateCommand command) {
        FixedAssetDO asset = repository.findById(command.getAssetId());
        if (asset == null) {
            throw new IllegalArgumentException("Asset not found");
        }

        FixedAssetDisposalDO disposal = new FixedAssetDisposalDO();
        disposal.setDisposalCode(command.getDisposalCode());
        disposal.setDisposalType(command.getDisposalType());
        disposal.setAssetId(command.getAssetId());
        disposal.setAssetCode(asset.getAssetCode());
        disposal.setAssetName(asset.getAssetName());
        disposal.setOriginalValue(asset.getOriginalValue());
        disposal.setNetValue(asset.getNetValue());
        disposal.setAccumulatedDepreciation(asset.getAccumulatedDepreciation());
        disposal.setDisposalAmount(command.getDisposalAmount());
        disposal.setDisposalDate(command.getDisposalDate());
        disposal.setDisposalReason(command.getDisposalReason());
        disposal.setDisposalMethod(command.getDisposalMethod());
        disposal.setAcquirerName(command.getAcquirerName());
        disposal.setAcquirerContact(command.getAcquirerContact());
        
        BigDecimal gainLoss = command.getDisposalAmount().subtract(asset.getNetValue());
        disposal.setGainLoss(gainLoss);
        
        disposal.setStatus("PENDING");
        disposal.setApprovalStatus("PENDING");
        disposal.setRemark(command.getRemark());
        disposal.setCreatedBy(command.getCreatedBy());
        disposal.setCreatorName(command.getCreatorName());
        disposal.setCreatedAt(LocalDateTime.now());
        disposal.setUpdatedAt(LocalDateTime.now());
        
        repository.saveDisposal(disposal);
        return disposal.getId();
    }

    public void approveDisposal(Long disposalId, Long approverId, String approverName, String comment) {
        FixedAssetDisposalDO disposal = repository.findDisposalByAssetId(disposalId).stream().findFirst().orElse(null);
        if (disposal != null) {
            disposal.setApprovalStatus("APPROVED");
            disposal.setApprovalComment(comment);
            disposal.setApproverId(approverId);
            disposal.setApproverName(approverName);
            disposal.setApprovalDate(java.time.LocalDate.now().toString());
            disposal.setStatus("APPROVED");
            disposal.setUpdatedAt(LocalDateTime.now());
            
            repository.saveDisposal(disposal);
            
            FixedAssetDO asset = repository.findById(disposal.getAssetId());
            if (asset != null) {
                asset.setStatus("DISPOSED");
                asset.setUsageStatus("DISPOSED");
                asset.setUpdatedAt(LocalDateTime.now());
                repository.save(asset);
            }
        }
    }

    public void rejectDisposal(Long disposalId, Long approverId, String approverName, String comment) {
        FixedAssetDisposalDO disposal = repository.findDisposalByAssetId(disposalId).stream().findFirst().orElse(null);
        if (disposal != null) {
            disposal.setApprovalStatus("REJECTED");
            disposal.setApprovalComment(comment);
            disposal.setApproverId(approverId);
            disposal.setApproverName(approverName);
            disposal.setApprovalDate(java.time.LocalDate.now().toString());
            disposal.setStatus("REJECTED");
            disposal.setUpdatedAt(LocalDateTime.now());
            repository.saveDisposal(disposal);
        }
    }
}