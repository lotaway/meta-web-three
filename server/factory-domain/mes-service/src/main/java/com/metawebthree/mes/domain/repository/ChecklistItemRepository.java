package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.ChecklistItem;
import java.util.List;
import java.util.Optional;

public interface ChecklistItemRepository {
    Optional<ChecklistItem> findById(Long id);
    Optional<ChecklistItem> findByItemCode(String itemCode);
    List<ChecklistItem> findByItemCategory(String itemCategory);
    List<ChecklistItem> findByStatus(ChecklistItem.ItemStatus status);
    List<ChecklistItem> findAll();
    ChecklistItem save(ChecklistItem item);
    void update(ChecklistItem item);
    void deleteById(Long id);
}