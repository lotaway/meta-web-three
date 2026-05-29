package com.metawebthree.finance.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.finance.domain.entity.AccountSubject;
import com.metawebthree.finance.domain.repository.AccountSubjectRepository;
import com.metawebthree.finance.infrastructure.persistence.converter.AccountSubjectConverter;
import com.metawebthree.finance.infrastructure.persistence.dataobject.AccountSubjectDO;
import com.metawebthree.finance.infrastructure.persistence.mapper.AccountSubjectMapper;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class AccountSubjectRepositoryImpl implements AccountSubjectRepository {

    private final AccountSubjectMapper subjectMapper;
    private final AccountSubjectConverter subjectConverter;

    public AccountSubjectRepositoryImpl(AccountSubjectMapper subjectMapper, 
                                         AccountSubjectConverter subjectConverter) {
        this.subjectMapper = subjectMapper;
        this.subjectConverter = subjectConverter;
    }

    @Override
    public Optional<AccountSubject> findById(Long id) {
        AccountSubjectDO subjectDO = subjectMapper.selectById(id);
        return Optional.ofNullable(subjectConverter.toEntity(subjectDO));
    }

    @Override
    public Optional<AccountSubject> findBySubjectCode(String subjectCode) {
        LambdaQueryWrapper<AccountSubjectDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(AccountSubjectDO::getSubjectCode, subjectCode);
        AccountSubjectDO subjectDO = subjectMapper.selectOne(wrapper);
        return Optional.ofNullable(subjectConverter.toEntity(subjectDO));
    }

    @Override
    public List<AccountSubject> findByParentId(Long parentId) {
        LambdaQueryWrapper<AccountSubjectDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(AccountSubjectDO::getParentId, parentId);
        List<AccountSubjectDO> subjectDOs = subjectMapper.selectList(wrapper);
        return subjectDOs.stream()
                .map(subjectConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<AccountSubject> findBySubjectCodeLike(String subjectCodePrefix) {
        LambdaQueryWrapper<AccountSubjectDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.likeRight(AccountSubjectDO::getSubjectCode, subjectCodePrefix);
        List<AccountSubjectDO> subjectDOs = subjectMapper.selectList(wrapper);
        return subjectDOs.stream()
                .map(subjectConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<AccountSubject> findByLevel(Integer level) {
        LambdaQueryWrapper<AccountSubjectDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(AccountSubjectDO::getLevel, level);
        List<AccountSubjectDO> subjectDOs = subjectMapper.selectList(wrapper);
        return subjectDOs.stream()
                .map(subjectConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<AccountSubject> findByStatus(AccountSubject.SubjectStatus status) {
        LambdaQueryWrapper<AccountSubjectDO> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(AccountSubjectDO::getStatus, status.name());
        List<AccountSubjectDO> subjectDOs = subjectMapper.selectList(wrapper);
        return subjectDOs.stream()
                .map(subjectConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public List<AccountSubject> findAll() {
        List<AccountSubjectDO> subjectDOs = subjectMapper.selectList(null);
        return subjectDOs.stream()
                .map(subjectConverter::toEntity)
                .collect(Collectors.toList());
    }

    @Override
    public void save(AccountSubject subject) {
        AccountSubjectDO subjectDO = subjectConverter.toDO(subject);
        if (subject.getId() == null) {
            subjectMapper.insert(subjectDO);
            subject.setId(subjectDO.getId());
        } else {
            subjectMapper.updateById(subjectDO);
        }
    }

    @Override
    public void update(AccountSubject subject) {
        AccountSubjectDO subjectDO = subjectConverter.toDO(subject);
        subjectMapper.updateById(subjectDO);
    }

    @Override
    public void delete(Long id) {
        subjectMapper.deleteById(id);
    }
}