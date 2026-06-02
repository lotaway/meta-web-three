package com.metawebthree.finance.infrastructure.persistence.repository.arap;

import com.metawebthree.finance.domain.entity.arap.AccountsPayable;
import com.metawebthree.finance.domain.entity.arap.AccountsPayable.ApStatus;
import com.metawebthree.finance.domain.repository.arap.AccountsPayableRepository;
import com.metawebthree.finance.infrastructure.persistence.converter.arap.AccountsPayableConverter;
import com.metawebthree.finance.infrastructure.persistence.dataobject.arap.AccountsPayableDO;
import com.metawebthree.finance.infrastructure.persistence.mapper.arap.AccountsPayableMapper;
import org.springframework.stereotype.Repository;

import java.time.LocalDate;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class AccountsPayableRepositoryImpl implements AccountsPayableRepository {

    private final AccountsPayableMapper apMapper;
    private final AccountsPayableConverter apConverter;

    public AccountsPayableRepositoryImpl(AccountsPayableMapper apMapper,
                                         AccountsPayableConverter apConverter) {
        this.apMapper = apMapper;
        this.apConverter = apConverter;
    }

    @Override
    public Optional<AccountsPayable> findById(Long id) {
        AccountsPayableDO apDO = apMapper.selectById(id);
        return Optional.ofNullable(apConverter.toEntity(apDO));
    }

    @Override
    public Optional<AccountsPayable> findByApCode(String apCode) {
        AccountsPayableDO apDO = apMapper.findByApCode(apCode);
        return Optional.ofNullable(apConverter.toEntity(apDO));
    }

    @Override
    public List<AccountsPayable> findBySupplierId(Long supplierId) {
        List<AccountsPayableDO> apDOs = apMapper.findBySupplierId(supplierId);
        return apDOs.stream()
            .map(apConverter::toEntity)
            .collect(Collectors.toList());
    }

    @Override
    public List<AccountsPayable> findByStatus(ApStatus status) {
        List<AccountsPayableDO> apDOs = apMapper.findByStatus(status.name());
        return apDOs.stream()
            .map(apConverter::toEntity)
            .collect(Collectors.toList());
    }

    @Override
    public List<AccountsPayable> findByDueDateBefore(LocalDate date) {
        List<AccountsPayableDO> apDOs = apMapper.findByDueDateBefore(date);
        return apDOs.stream()
            .map(apConverter::toEntity)
            .collect(Collectors.toList());
    }

    @Override
    public List<AccountsPayable> findBySupplierIdAndStatus(Long supplierId, ApStatus status) {
        List<AccountsPayableDO> apDOs = apMapper.findBySupplierIdAndStatus(supplierId, status.name());
        return apDOs.stream()
            .map(apConverter::toEntity)
            .collect(Collectors.toList());
    }

    @Override
    public List<AccountsPayable> findAll() {
        List<AccountsPayableDO> apDOs = apMapper.selectList(null);
        return apDOs.stream()
            .map(apConverter::toEntity)
            .collect(Collectors.toList());
    }

    @Override
    public AccountsPayable save(AccountsPayable ap) {
        AccountsPayableDO apDO = apConverter.toDO(ap);
        if (ap.getId() == null) {
            apMapper.insert(apDO);
            ap.setId(apDO.getId());
        } else {
            apMapper.updateById(apDO);
        }
        return ap;
    }

    @Override
    public void delete(AccountsPayable ap) {
        if (ap.getId() != null) {
            apMapper.deleteById(ap.getId());
        }
    }
}