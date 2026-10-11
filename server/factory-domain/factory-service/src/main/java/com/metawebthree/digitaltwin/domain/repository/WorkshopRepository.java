package com.metawebthree.digitaltwin.domain.repository;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.metawebthree.digitaltwin.domain.entity.Workshop;
import java.util.List;
import java.util.Optional;

public interface WorkshopRepository {
    Optional<Workshop> findById(Long id);
    Optional<Workshop> findByWorkshopCode(String workshopCode);
    List<Workshop> findByStatus(Workshop.WorkshopStatus status);
    List<Workshop> findAll();
    IPage<Workshop> findPaginated(int page, int size);
    Workshop save(Workshop workshop);
    void update(Workshop workshop);
    void deleteById(Long id);
}