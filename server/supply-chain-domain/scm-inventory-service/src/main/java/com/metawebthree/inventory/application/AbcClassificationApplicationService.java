package com.metawebthree.inventory.application;

import com.metawebthree.inventory.application.dto.AbcClassificationDTO;
import java.util.List;

public interface AbcClassificationApplicationService {
    List<AbcClassificationDTO> classify(Long warehouseId, Integer periodDays);
}