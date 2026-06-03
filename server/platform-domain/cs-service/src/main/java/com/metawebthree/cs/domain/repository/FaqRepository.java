package com.metawebthree.cs.domain.repository;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.cs.domain.model.Faq;

import java.util.List;

public interface FaqRepository {
    Faq save(Faq faq);
    
    Faq findById(Long id);
    
    List<Faq> findAll();
    
    IPage<Faq> findAllPaged(Page<Faq> page);
    
    List<Faq> findByCategory(String category);
    
    IPage<Faq> findByCategoryPaged(Page<Faq> page, String category);
    
    List<Faq> findByEnabled(Boolean enabled);
    
    List<Faq> searchByKeyword(String keyword);
    
    IPage<Faq> searchByKeywordPaged(Page<Faq> page, String keyword);
    
    List<Faq> findTopByRelevance(int limit);
    
    void deleteById(Long id);
    
    Long count();
}