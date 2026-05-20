package com.metawebthree.supplier.application;

import com.metawebthree.supplier.application.dto.SupplierDTO;
import com.metawebthree.supplier.domain.entity.Supplier;
import com.metawebthree.supplier.domain.repository.SupplierRepository;
import org.springframework.stereotype.Service;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class SupplierApplicationServiceImpl implements SupplierApplicationService {

    private final SupplierRepository repository;

    public SupplierApplicationServiceImpl(SupplierRepository repository) {
        this.repository = repository;
    }

    @Override
    public SupplierDTO createSupplier(SupplierDTO dto) {
        Supplier supplier = new Supplier();
        supplier.setSupplierCode(dto.getSupplierCode());
        supplier.setName(dto.getName());
        supplier.setCategory(dto.getCategory());
        supplier.setContactPerson(dto.getContactPerson());
        supplier.setContactPhone(dto.getContactPhone());
        supplier.setAddress(dto.getAddress());
        supplier.setStatus("ACTIVE");
        supplier.setAssessmentLevel(dto.getAssessmentLevel() != null ? dto.getAssessmentLevel() : "B");
        
        Supplier saved = repository.save(supplier);
        return toDTO(saved);
    }

    @Override
    public SupplierDTO updateSupplier(Long id, SupplierDTO dto) {
        return repository.findById(id)
            .map(s -> {
                s.setName(dto.getName());
                s.setCategory(dto.getCategory());
                s.setContactPerson(dto.getContactPerson());
                s.setContactPhone(dto.getContactPhone());
                s.setAddress(dto.getAddress());
                s.setAssessmentLevel(dto.getAssessmentLevel());
                return toDTO(repository.save(s));
            })
            .orElse(null);
    }

    @Override
    public SupplierDTO querySupplier(Long id) {
        return repository.findById(id).map(this::toDTO).orElse(null);
    }

    @Override
    public SupplierDTO queryByCode(String supplierCode) {
        return repository.findByCode(supplierCode).map(this::toDTO).orElse(null);
    }

    @Override
    public List<SupplierDTO> listSuppliers(String status, String category) {
        List<Supplier> suppliers;
        if (status != null) {
            suppliers = repository.findByStatus(status);
        } else if (category != null) {
            suppliers = repository.findByCategory(category);
        } else {
            suppliers = List.copyOf(repository.findByStatus("ACTIVE"));
        }
        return suppliers.stream().map(this::toDTO).collect(Collectors.toList());
    }

    public SupplierDTO updateAssessment(Long id, String level) {
        return repository.findById(id)
            .map(s -> {
                s.setAssessmentLevel(level);
                return toDTO(repository.save(s));
            })
            .orElse(null);
    }

    private SupplierDTO toDTO(Supplier s) {
        SupplierDTO dto = new SupplierDTO();
        dto.setId(s.getId());
        dto.setSupplierCode(s.getSupplierCode());
        dto.setName(s.getName());
        dto.setCategory(s.getCategory());
        dto.setContactPerson(s.getContactPerson());
        dto.setContactPhone(s.getContactPhone());
        dto.setAddress(s.getAddress());
        dto.setStatus(s.getStatus());
        dto.setAssessmentLevel(s.getAssessmentLevel());
        dto.setCreatedAt(s.getCreatedAt());
        dto.setUpdatedAt(s.getUpdatedAt());
        return dto;
    }
}