package com.metawebthree.mes.infrastructure.persistence.repository;

import com.metawebthree.mes.domain.entity.DataDictionary;
import com.metawebthree.mes.domain.repository.DataDictionaryRepository;
import org.springframework.stereotype.Repository;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

@Repository
public class DataDictionaryRepositoryImpl implements DataDictionaryRepository {
    
    private final Map<Long, DataDictionary> storage = new ConcurrentHashMap<>();
    private final AtomicLong idGen = new AtomicLong(1);
    
    @Override
    public Optional<DataDictionary> findById(Long id) {
        return Optional.ofNullable(storage.get(id));
    }
    
    @Override
    public Optional<DataDictionary> findByDictCode(String dictCode) {
        return storage.values().stream()
                .filter(d -> d.getDictCode().equals(dictCode))
                .findFirst();
    }
    
    @Override
    public List<DataDictionary> findAllActive() {
        return storage.values().stream()
                .filter(d -> d.getStatus() == DataDictionary.DictStatus.ACTIVE)
                .sorted(Comparator.comparing(DataDictionary::getSortOrder, Comparator.nullsLast(Comparator.naturalOrder())))
                .collect(Collectors.toList());
    }
    
    @Override
    public DataDictionary save(DataDictionary dictionary) {
        if (dictionary.getId() == null) {
            dictionary.setId(idGen.getAndIncrement());
        }
        
        // 如果有字典项，更新它们的dictId
        if (dictionary.getItems() != null) {
            for (DataDictionary.DataDictionaryItem item : dictionary.getItems()) {
                if (item.getDictId() == null) {
                    item.setDictId(dictionary.getId());
                }
            }
        }
        
        storage.put(dictionary.getId(), dictionary);
        return dictionary;
    }
    
    @Override
    public void delete(Long id) {
        storage.remove(id);
    }
    
    @Override
    public boolean existsByDictCode(String dictCode) {
        return storage.values().stream()
                .anyMatch(d -> d.getDictCode().equals(dictCode));
    }
}