package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.DataDictionary;
import java.util.List;
import java.util.Optional;

/**
 * 数据字典仓储接口
 */
public interface DataDictionaryRepository {
    
    /**
     * 根据ID查询
     */
    Optional<DataDictionary> findById(Long id);
    
    /**
     * 根据字典编码查询
     */
    Optional<DataDictionary> findByDictCode(String dictCode);
    
    /**
     * 查询所有启用的字典
     */
    List<DataDictionary> findAllActive();
    
    /**
     * 保存数据字典
     */
    DataDictionary save(DataDictionary dictionary);
    
    /**
     * 删除数据字典
     */
    void delete(Long id);
    
    /**
     * 检查字典编码是否已存在
     */
    boolean existsByDictCode(String dictCode);
}