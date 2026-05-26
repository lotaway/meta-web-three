package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.DataDictionary;
import java.util.List;
import java.util.Optional;

public interface DataDictionaryRepository {
    
    Optional<DataDictionary> findById(Long id);
    
    Optional<DataDictionary> findByDictCode(String dictCode);
    
    List<DataDictionary> findAllActive();
    
    DataDictionary save(DataDictionary dictionary);
    
    void delete(Long id);
    
    boolean existsByDictCode(String dictCode);
}