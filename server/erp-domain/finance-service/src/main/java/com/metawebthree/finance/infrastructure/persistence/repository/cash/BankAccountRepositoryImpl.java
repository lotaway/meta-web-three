package com.metawebthree.finance.infrastructure.persistence.repository.cash;

import com.metawebthree.finance.domain.entity.cash.BankAccount;
import com.metawebthree.finance.domain.repository.cash.BankAccountRepository;
import com.metawebthree.finance.infrastructure.persistence.converter.CashConverter;
import com.metawebthree.finance.infrastructure.persistence.mapper.cash.BankAccountMapper;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public class BankAccountRepositoryImpl implements BankAccountRepository {
    private final BankAccountMapper bankAccountMapper;
    private final CashConverter converter;

    public BankAccountRepositoryImpl(BankAccountMapper bankAccountMapper, CashConverter converter) {
        this.bankAccountMapper = bankAccountMapper;
        this.converter = converter;
    }

    @Override
    public Long save(BankAccount bankAccount) {
        var doObj = converter.toDO(bankAccount);
        bankAccountMapper.insert(doObj);
        return doObj.getId();
    }

    @Override
    public void update(BankAccount bankAccount) {
        var doObj = converter.toDO(bankAccount);
        bankAccountMapper.updateById(doObj);
    }

    @Override
    public Optional<BankAccount> findById(Long id) {
        var doObj = bankAccountMapper.selectById(id);
        return Optional.ofNullable(doObj).map(converter::toEntity);
    }

    @Override
    public Optional<BankAccount> findByAccountCode(String accountCode) {
        var doObj = bankAccountMapper.selectOne(
            new com.baomidou.mybatisplus.core.conditions.query.QueryWrapper<com.metawebthree.finance.infrastructure.persistence.dataobject.cash.BankAccountDO>()
                .eq("account_code", accountCode)
        );
        return Optional.ofNullable(doObj).map(converter::toEntity);
    }

    @Override
    public List<BankAccount> findAll() {
        return bankAccountMapper.selectList(null).stream()
            .map(converter::toEntity)
            .toList();
    }

    @Override
    public List<BankAccount> findByStatus(BankAccount.BankAccountStatus status) {
        return bankAccountMapper.selectList(
            new com.baomidou.mybatisplus.core.conditions.query.QueryWrapper<com.metawebthree.finance.infrastructure.persistence.dataobject.cash.BankAccountDO>()
                .eq("status", status.name())
        ).stream().map(converter::toEntity).toList();
    }

    @Override
    public List<BankAccount> findByIsActive(Boolean isActive) {
        return bankAccountMapper.selectList(
            new com.baomidou.mybatisplus.core.conditions.query.QueryWrapper<com.metawebthree.finance.infrastructure.persistence.dataobject.cash.BankAccountDO>()
                .eq("is_active", isActive)
        ).stream().map(converter::toEntity).toList();
    }

    @Override
    public void deleteById(Long id) {
        bankAccountMapper.deleteById(id);
    }
}