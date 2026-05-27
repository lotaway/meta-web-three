package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.SopDocument;
import com.metawebthree.mes.domain.entity.SopRouteBinding;
import java.util.List;

public interface SopRouteBindingRepository {
    List<SopDocument> findByRouteCodeAndStepNo(String routeCode, Integer stepNo);
    List<SopDocument> findByWorkstationId(String workstationId);
    List<SopRouteBinding> findBySopDocumentId(Long sopDocumentId);
    void save(SopRouteBinding binding);
    void deleteBySopDocumentId(Long sopDocumentId);
}