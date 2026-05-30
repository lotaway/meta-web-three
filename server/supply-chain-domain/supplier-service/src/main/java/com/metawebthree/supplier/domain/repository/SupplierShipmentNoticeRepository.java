package com.metawebthree.supplier.domain.repository;

import com.metawebthree.supplier.domain.entity.SupplierShipmentNotice;
import java.util.List;
import java.util.Optional;

public interface SupplierShipmentNoticeRepository {
    
    SupplierShipmentNotice save(SupplierShipmentNotice notice);
    
    Optional<SupplierShipmentNotice> findById(Long id);
    
    Optional<SupplierShipmentNotice> findByNoticeNo(String noticeNo);
    
    List<SupplierShipmentNotice> findBySupplierCode(String supplierCode);
    
    List<SupplierShipmentNotice> findByOrderNo(String orderNo);
    
    List<SupplierShipmentNotice> findBySupplierCodeAndStatus(String supplierCode, String status);
    
    List<SupplierShipmentNotice> findBySupplierCodeAndStatusIn(String supplierCode, List<String> statuses);
    
    void deleteById(Long id);
}