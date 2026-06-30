package com.metawebthree.supplier.infrastructure.persistence.repository;

import com.metawebthree.supplier.domain.entity.Supplier;
import com.metawebthree.supplier.domain.repository.SupplierRepository;
import com.metawebthree.supplier.infrastructure.persistence.dataobject.SupplierDO;
import com.metawebthree.supplier.infrastructure.persistence.mapper.SupplierMapper;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
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
        SupplierDO suppDO = toDO(supplier);
        if (supplier.getId() == null) {
            suppDO.setCreatedAt(LocalDateTime.now());
            supplierMapper.insert(suppDO);
            supplier.setId(suppDO.getId());
        } else {
            suppDO.setUpdatedAt(LocalDateTime.now());
            supplierMapper.updateById(suppDO);
        }
        return supplier;
    }

    @Override
    public Optional<Supplier> findById(Long id) {
        SupplierDO suppDO = supplierMapper.selectById(id);
        return Optional.ofNullable(toEntity(suppDO));
    }

    @Override
    public Optional<Supplier> findByCode(String supplierCode) {
        SupplierDO suppDO = supplierMapper.selectByCode(supplierCode);
        return Optional.ofNullable(toEntity(suppDO));
    }

    @Override
    public List<Supplier> findByStatus(String status) {
        List<SupplierDO> list = supplierMapper.selectByStatus(status);
        return list.stream().map(this::toEntity).toList();
    }

    @Override
    public List<Supplier> findByCategory(String category) {
        List<SupplierDO> list = supplierMapper.selectByCategory(category);
        return list.stream().map(this::toEntity).toList();
    }

    @Override
    public List<Supplier> findByAssessmentLevel(String level) {
        List<SupplierDO> list = supplierMapper.selectByAssessmentLevel(level);
        return list.stream().map(this::toEntity).toList();
    }

    private SupplierDO toDO(Supplier supplier) {
        SupplierDO suppDO = new SupplierDO();
        suppDO.setId(supplier.getId());
        suppDO.setSupplierCode(supplier.getSupplierCode());
        suppDO.setSupplierName(supplier.getSupplierName());
        suppDO.setSupplierType(supplier.getSupplierType());
        suppDO.setBusinessLicense(supplier.getBusinessLicense());
        suppDO.setTaxId(supplier.getTaxId());
        suppDO.setProvince(supplier.getProvince());
        suppDO.setCity(supplier.getCity());
        suppDO.setDistrict(supplier.getDistrict());
        suppDO.setAddress(supplier.getAddress());
        suppDO.setContact(supplier.getContact());
        suppDO.setPhone(supplier.getPhone());
        suppDO.setEmail(supplier.getEmail());
        suppDO.setStatus(supplier.getStatus());
        suppDO.setCreditLimit(supplier.getCreditLimit());
        suppDO.setPaymentTerms(supplier.getPaymentTerms());
        suppDO.setCategory(supplier.getCategory());
        suppDO.setScore(supplier.getScore());
        suppDO.setLevel(supplier.getLevel());
        suppDO.setAssessmentLevel(supplier.getAssessmentLevel());
        suppDO.setContactPerson(supplier.getContactPerson());
        suppDO.setContactPhone(supplier.getContactPhone());
        suppDO.setCreatedAt(supplier.getCreatedAt());
        suppDO.setUpdatedAt(supplier.getUpdatedAt());
        return suppDO;
    }

    private Supplier toEntity(SupplierDO suppDO) {
        if (suppDO == null) {
            return null;
        }
        Supplier supplier = new Supplier();
        supplier.setId(suppDO.getId());
        supplier.setSupplierCode(suppDO.getSupplierCode());
        supplier.setSupplierName(suppDO.getSupplierName());
        supplier.setSupplierType(suppDO.getSupplierType());
        supplier.setBusinessLicense(suppDO.getBusinessLicense());
        supplier.setTaxId(suppDO.getTaxId());
        supplier.setProvince(suppDO.getProvince());
        supplier.setCity(suppDO.getCity());
        supplier.setDistrict(suppDO.getDistrict());
        supplier.setAddress(suppDO.getAddress());
        supplier.setContact(suppDO.getContact());
        supplier.setPhone(suppDO.getPhone());
        supplier.setEmail(suppDO.getEmail());
        supplier.setStatus(suppDO.getStatus());
        supplier.setCreditLimit(suppDO.getCreditLimit());
        supplier.setPaymentTerms(suppDO.getPaymentTerms());
        supplier.setCategory(suppDO.getCategory());
        supplier.setScore(suppDO.getScore());
        supplier.setLevel(suppDO.getLevel());
        supplier.setAssessmentLevel(suppDO.getAssessmentLevel());
        supplier.setContactPerson(suppDO.getContactPerson());
        supplier.setContactPhone(suppDO.getContactPhone());
        supplier.setCreatedAt(suppDO.getCreatedAt());
        supplier.setUpdatedAt(suppDO.getUpdatedAt());
        return supplier;
    }
}