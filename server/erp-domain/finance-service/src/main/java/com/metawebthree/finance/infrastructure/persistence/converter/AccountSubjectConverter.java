package com.metawebthree.finance.infrastructure.persistence.converter;

import com.metawebthree.finance.domain.entity.AccountSubject;
import com.metawebthree.finance.infrastructure.persistence.dataobject.AccountSubjectDO;
import org.springframework.stereotype.Component;

@Component
public class AccountSubjectConverter {

    public AccountSubject toEntity(AccountSubjectDO doObj) {
        if (doObj == null) {
            return null;
        }
        AccountSubject entity = new AccountSubject();
        entity.setId(doObj.getId());
        entity.setSubjectCode(doObj.getSubjectCode());
        entity.setSubjectName(doObj.getSubjectName());
        entity.setDirection(AccountSubject.SubjectDirection.valueOf(doObj.getDirection()));
        entity.setParentId(doObj.getParentId());
        entity.setLevel(doObj.getLevel());
        entity.setStatus(AccountSubject.SubjectStatus.valueOf(doObj.getStatus()));
        entity.setBalance(doObj.getBalance());
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setUpdatedAt(doObj.getUpdatedAt());
        return entity;
    }

    public AccountSubjectDO toDO(AccountSubject entity) {
        if (entity == null) {
            return null;
        }
        AccountSubjectDO doObj = new AccountSubjectDO();
        doObj.setId(entity.getId());
        doObj.setSubjectCode(entity.getSubjectCode());
        doObj.setSubjectName(entity.getSubjectName());
        doObj.setDirection(entity.getDirection() != null ? entity.getDirection().name() : null);
        doObj.setParentId(entity.getParentId());
        doObj.setLevel(entity.getLevel());
        doObj.setStatus(entity.getStatus() != null ? entity.getStatus().name() : null);
        doObj.setBalance(entity.getBalance());
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setUpdatedAt(entity.getUpdatedAt());
        return doObj;
    }
}