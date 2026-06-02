package com.metawebthree.finance.infrastructure.persistence.repository.arap;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.finance.domain.entity.arap.AccountsReceivable;
import com.metawebthree.finance.domain.entity.arap.AccountsReceivable.ArStatus;
import com.metawebthree.finance.domain.repository.arap.AccountsReceivableRepository;
import com.metawebthree.finance.infrastructure.persistence.converter.arap.AccountsReceivableConverter;
import com.metawebthree.finance.infrastructure.persistence.dataobject.arap.AccountsReceivableDO;
import com.metawebthree.finance.infrastructure.persistence.mapper.arap.AccountsReceivableMapper;
import org.springframework.stereotype.Repository;

import java.time.LocalDate;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class AccountsReceivableRepositoryImpl implements AccountsReceivableRepository {

    private final AccountsReceivableMapper arMapper;
    private final AccountsReceivableConverter arConverter;

    public AccountsReceivableRepositoryImpl(AccountsReceivableMapper arMapper,
                                            AccountsReceivableConverter arConverter) {
        this.arMapper = arMapper;
        this.arConverter = arConverter;
    }

    @Override
    public Optional<AccountsReceivable> findById(Long id) {
        AccountsReceivableDO arDO = arMapper.selectById(id);
        return Optional.ofNullable(arConverter.toEntity(arDO));
    }

    @Override
    public Optional<AccountsReceivable> findByArCode(String arCode) {
        AccountsReceivableDO arDO = arMapper.findByArCode(arCode);
        return Optional.ofNullable(arConverter.toEntity(arDO));
    }

    @Override
    public List<AccountsReceivable> findByCustomerId(Long customerId) {
        List<AccountsReceivableDO> arDOs = arMapper.findByCustomerId(customerId);
        return arDOs.stream()
            .map(arConverter::toEntity)
            .collect(Collectors.toList());
    }

    @Override
    public List<AccountsReceivable> findByStatus(ArStatus status) {
        List<AccountsReceivableDO> arDOs = arMapper.findByStatus(status.name());
        return arDOs.stream()
            .map(arConverter::toEntity)
            .collect(Collectors.toList());
    }

    @Override
    public List<AccountsReceivable> findByDueDateBefore(LocalDate date) {
        List<AccountsReceivableDO> arDOs = arMapper.findByDueDateBefore(date);
        return arDOs.stream()
            .map(arConverter::toEntity)
            .collect(Collectors.toList());
    }

    @Override
    public List<AccountsReceivable> findByCustomerIdAndStatus(Long customerId, ArStatus status) {
        List<AccountsReceivableDO> arDOs = arMapper.findByCustomerIdAndStatus(customerId, status.name());
        return arDOs.stream()
            .map(arConverter::toEntity)
            .collect(Collectors.toList());
    }

    @Override
    public List<AccountsReceivable> findAll() {
        List<AccountsReceivableDO> arDOs = arMapper.selectList(null);
        return arDOs.stream()
            .map(arConverter::toEntity)
            .collect(Collectors.toList());
    }

    @Override
    public AccountsReceivable save(AccountsReceivable ar) {
        AccountsReceivableDO arDO = arConverter.toDO(ar);
        if (ar.getId() == null) {
            arMapper.insert(arDO);
            ar.setId(arDO.getId());
        } else {
            arMapper.updateById(arDO);
        }
        return ar;
    }

    @Override
    public void delete(AccountsReceivable ar) {
        if (ar.getId() != null) {
            arMapper.deleteById(ar.getId());
        }
    }
}