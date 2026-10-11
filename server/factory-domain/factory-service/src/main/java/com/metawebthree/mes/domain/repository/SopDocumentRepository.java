package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.SopDocument;

import java.util.List;
import java.util.Optional;

public interface SopDocumentRepository {
    
    SopDocument save(SopDocument sopDocument);
    
    Optional<SopDocument> findById(Long id);
    
    Optional<SopDocument> findByDocumentCode(String documentCode);
    
    List<SopDocument> findAll();
    
    List<SopDocument> findByStatus(SopDocument.SopStatus status);
    
    List<SopDocument> findByDocumentType(String documentType);
    
    List<SopDocument> findByCategory(String category);
    
    boolean existsByDocumentCode(String documentCode);
    
    void deleteById(Long id);
    
    List<SopDocument> findByDocumentNameContaining(String documentName);
}