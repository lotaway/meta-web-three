package com.metawebthree.supplier.application;

import com.metawebthree.supplier.application.dto.SupplierDTO;
import org.springframework.stereotype.Service;
import java.util.List;

@Service
public class SupplierServiceRpcImpl {

    private final SupplierApplicationService appService;

    public SupplierServiceRpcImpl(SupplierApplicationService appService) {
        this.appService = appService;
    }

    public SupplierDTO getSupplier(Long id) {
        return appService.querySupplier(id);
    }

    public SupplierDTO getByCode(String supplierCode) {
        return appService.queryByCode(supplierCode);
    }

    public List<SupplierDTO> listSuppliers(String status, String category) {
        return appService.listSuppliers(status, category);
    }
}