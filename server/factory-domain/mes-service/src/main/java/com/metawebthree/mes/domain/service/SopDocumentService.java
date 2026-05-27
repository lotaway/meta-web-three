package com.metawebthree.mes.domain.service;

import com.metawebthree.mes.domain.entity.SopDocument;
import com.metawebthree.mes.domain.entity.SopDocumentVersion;

import java.util.List;
import java.util.Optional;

public interface SopDocumentService {
    
    SopDocument create(SopDocument sopDocument);
    
    Optional<SopDocument> findById(Long id);
    
    Optional<SopDocument> findByDocumentCode(String documentCode);
    
    List<SopDocument> findAll();
    
    List<SopDocument> findByStatus(SopDocument.SopStatus status);
    
    List<SopDocument> findByDocumentType(String documentType);
    
    SopDocument update(SopDocument sopDocument);
    
    void deleteById(Long id);
    
    SopDocument activate(Long id);
    
    SopDocument archive(Long id);
    
    SopDocumentVersion addVersion(Long sopDocumentId, String fileName, 
                                               String filePath, String uploader, 
                                               String changeDescription);
    
    SopDocument bindRoute(Long sopDocumentId, String routeCode, String routeName, 
                          Integer stepNo, String processCode, String processName,
                          String workstationId, String workstationName);
    
    SopDocument unbindRoute(Long sopDocumentId, String routeCode, Integer stepNo);
    
    List<SopDocument> findByRouteAndStep(String routeCode, Integer stepNo);
    
    List<SopDocument> findByWorkstation(String workstationId);
}