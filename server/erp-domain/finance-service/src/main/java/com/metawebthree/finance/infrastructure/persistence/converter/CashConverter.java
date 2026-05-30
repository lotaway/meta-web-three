package com.metawebthree.finance.infrastructure.persistence.converter;

import com.metawebthree.finance.domain.entity.cash.BankAccount;
import com.metawebthree.finance.infrastructure.persistence.dataobject.cash.BankAccountDO;
import org.springframework.stereotype.Component;

@Component
public class CashConverter {

    public BankAccount toEntity(BankAccountDO doObj) {
        if (doObj == null) {
            return null;
        }
        BankAccount entity = new BankAccount();
        entity.setId(doObj.getId());
        entity.setAccountCode(doObj.getAccountCode());
        entity.setAccountName(doObj.getAccountName());
        entity.setBankName(doObj.getBankName());
        entity.setAccountNumber(doObj.getAccountNumber());
        entity.setAccountType(doObj.getAccountType());
        entity.setStatus(doObj.getStatus() != null ? BankAccount.BankAccountStatus.valueOf(doObj.getStatus()) : null);
        entity.setBalance(doObj.getBalance());
        entity.setFrozenAmount(doObj.getFrozenAmount());
        entity.setCurrency(doObj.getCurrency());
        entity.setRemark(doObj.getRemark());
        entity.setCreatedBy(doObj.getCreatedBy());
        entity.setCreatorName(doObj.getCreatorName());
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        entity.setIsActive(doObj.getIsActive());
        return entity;
    }

    public BankAccountDO toDO(BankAccount entity) {
        if (entity == null) {
            return null;
        }
        BankAccountDO doObj = new BankAccountDO();
        doObj.setId(entity.getId());
        doObj.setAccountCode(entity.getAccountCode());
        doObj.setAccountName(entity.getAccountName());
        doObj.setBankName(entity.getBankName());
        doObj.setAccountNumber(entity.getAccountNumber());
        doObj.setAccountType(entity.getAccountType());
        doObj.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        doObj.setBalance(entity.getBalance());
        doObj.setFrozenAmount(entity.getFrozenAmount());
        doObj.setCurrency(entity.getCurrency());
        doObj.setRemark(entity.getRemark());
        doObj.setCreatedBy(entity.getCreatedBy());
        doObj.setCreatorName(entity.getCreatorName());
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        doObj.setIsActive(entity.getIsActive());
        return doObj;
    }
}