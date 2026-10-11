package com.metawebthree.supplier.infrastructure.persistence.repository;

import com.metawebthree.supplier.domain.model.Supplier;
import com.metawebthree.supplier.domain.repository.SupplierRepository;
import com.metawebthree.supplier.infrastructure.persistence.entity.SupplierEntity;
import com.metawebthree.supplier.infrastructure.persistence.mapper.SupplierMapper;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public class SupplierRepositoryImpl implements SupplierRepository {

    private final SupplierMapper supplierMapper;

    public SupplierRepositoryImpl(SupplierMapper supplierMapper) {
        this.supplierMapper = supplierMapper;
    }

    @Override
    public Supplier save(Supplier supplier) {
        SupplierEntity entity = toEntity(supplier);
        if (supplier.getId() == null) {
            supplierMapper.insert(entity);
        } else {
            supplierMapper.updateById(entity);
        }
        return toDomain(entity);
    }

    @Override
    public Optional<Supplier> findById(Long id) {
        SupplierEntity entity = supplierMapper.selectById(id);
        return Optional.ofNullable(toDomain(entity));
    }

    @Override
    public Optional<Supplier> findBySupplierCode(String supplierCode) {
        SupplierEntity entity = supplierMapper.selectOne(
                new com.baomidou.mybatisplus.core.conditions.query.QueryWrapper<SupplierEntity>()
                        .eq("supplier_code", supplierCode)
        );
        return Optional.ofNullable(toDomain(entity));
    }

    @Override
    public List<Supplier> findAll() {
        return supplierMapper.selectList(null).stream()
                .map(this::toDomain)
                .collect(java.util.stream.Collectors.toList());
    }

    @Override
    public List<Supplier> findByStatus(Supplier.SupplierStatus status) {
        return supplierMapper.selectList(
                new com.baomidou.mybatisplus.core.conditions.query.QueryWrapper<SupplierEntity>()
                        .eq("status", status.getValue())
        ).stream()
                .map(this::toDomain)
                .collect(java.util.stream.Collectors.toList());
    }

    @Override
    public List<Supplier> findByVerificationStatus(Supplier.VerificationStatus verificationStatus) {
        return supplierMapper.selectList(
                new com.baomidou.mybatisplus.core.conditions.query.QueryWrapper<SupplierEntity>()
                        .eq("verification_status", verificationStatus.getValue())
        ).stream()
                .map(this::toDomain)
                .collect(java.util.stream.Collectors.toList());
    }

    @Override
    public boolean deleteById(Long id) {
        return supplierMapper.deleteById(id) > 0;
    }

    @Override
    public List<Supplier> findBySupplierLevel(Integer level) {
        return supplierMapper.selectList(
                new com.baomidou.mybatisplus.core.conditions.query.QueryWrapper<SupplierEntity>()
                        .eq("supplier_level", level)
        ).stream()
                .map(this::toDomain)
                .collect(java.util.stream.Collectors.toList());
    }

    private SupplierEntity toEntity(Supplier supplier) {
        SupplierEntity entity = new SupplierEntity();
        entity.setId(supplier.getId());
        entity.setSupplierCode(supplier.getSupplierCode());
        entity.setSupplierName(supplier.getSupplierName());
        entity.setContactPerson(supplier.getContactPerson());
        entity.setContactPhone(supplier.getContactPhone());
        entity.setContactEmail(supplier.getContactEmail());
        entity.setAddress(supplier.getAddress());
        entity.setStatus(supplier.getStatus() != null ? supplier.getStatus().getValue() : null);
        entity.setVerificationStatus(supplier.getVerificationStatus() != null ? supplier.getVerificationStatus().getValue() : null);
        entity.setBusinessLicense(supplier.getBusinessLicense());
        entity.setLegalPerson(supplier.getLegalPerson());
        entity.setSupplierLevel(supplier.getSupplierLevel());
        entity.setScore(supplier.getScore());
        entity.setRemark(supplier.getRemark());
        entity.setCreateTime(supplier.getCreateTime());
        entity.setUpdateTime(supplier.getUpdateTime());
        return entity;
    }

    private Supplier toDomain(SupplierEntity entity) {
        if (entity == null) {
            return null;
        }
        Supplier supplier = new Supplier();
        supplier.setId(entity.getId());
        supplier.setSupplierCode(entity.getSupplierCode());
        supplier.setSupplierName(entity.getSupplierName());
        supplier.setContactPerson(entity.getContactPerson());
        supplier.setContactPhone(entity.getContactPhone());
        supplier.setContactEmail(entity.getContactEmail());
        supplier.setAddress(entity.getAddress());
        supplier.setStatus(Supplier.SupplierStatus.fromValue(entity.getStatus()));
        supplier.setVerificationStatus(Supplier.VerificationStatus.fromValue(entity.getVerificationStatus()));
        supplier.setBusinessLicense(entity.getBusinessLicense());
        supplier.setLegalPerson(entity.getLegalPerson());
        supplier.setSupplierLevel(entity.getSupplierLevel());
        supplier.setScore(entity.getScore());
        supplier.setRemark(entity.getRemark());
        supplier.setCreateTime(entity.getCreateTime());
        supplier.setUpdateTime(entity.getUpdateTime());
        return supplier;
    }
}