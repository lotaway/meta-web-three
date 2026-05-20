package com.metawebthree.supplier.application;

import com.metawebthree.supplier.application.dto.SupplierDTO;
import java.util.List;

public interface SupplierApplicationService {

    SupplierDTO createSupplier(SupplierDTO dto);

    SupplierDTO updateSupplier(Long id, SupplierDTO dto);

    SupplierDTO querySupplier(Long id);

    SupplierDTO queryByCode(String supplierCode);

    List<SupplierDTO> listSuppliers(String status, String category);

    SupplierDTO updateAssessment(Long id, String level);
}