package com.metawebthree.finance.infrastructure.persistence.converter;

import com.metawebthree.finance.domain.entity.Account;
import com.metawebthree.finance.infrastructure.persistence.dataobject.AccountDO;
import org.springframework.stereotype.Component;


@Component
public class AccountConverter {

    public Account toEntity(AccountDO doObj) {
        if (doObj == null) {
            return null;
        }
        Account entity = new Account();
        entity.setId(doObj.getId());
        entity.setAccountNo(doObj.getAccountNo());
        entity.setAccountName(doObj.getAccountName());
        entity.setType(Account.AccountType.valueOf(doObj.getType()));
        entity.setBalance(doObj.getBalance());
        entity.setStatus(Account.AccountStatus.valueOf(doObj.getStatus()));
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        return entity;
    }

    public AccountDO toDO(Account entity) {
        if (entity == null) {
            return null;
        }
        AccountDO doObj = new AccountDO();
        doObj.setId(entity.getId());
        doObj.setAccountNo(entity.getAccountNo());
        doObj.setAccountName(entity.getAccountName());
        doObj.setType(entity.getType() != null ? entity.getType().name() : null);
        doObj.setBalance(entity.getBalance());
        doObj.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        return doObj;
    }
}