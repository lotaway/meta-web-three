package com.metawebthree.cs.domain.repository;

import com.metawebthree.cs.domain.model.Faq;

import java.util.List;

public interface FaqRepository {
    Faq save(Faq faq);
    
    Faq findById(Long id);
    
    List<Faq> findAll();
    
    List<Faq> findByCategory(String category);
    
    List<Faq> findByEnabled(Boolean enabled);
    
    List<Faq> searchByKeyword(String keyword);
    
    List<Faq> findTopByRelevance(int limit);
    
    void deleteById(Long id);
    
    Long count();
}