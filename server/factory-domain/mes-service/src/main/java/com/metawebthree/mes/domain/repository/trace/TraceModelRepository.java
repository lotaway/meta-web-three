package com.metawebthree.mes.domain.repository.trace;

import com.metawebthree.mes.domain.entity.TraceModel;
import java.util.List;
import java.util.Optional;

public interface TraceModelRepository {
    Optional<TraceModel> findById(Long id);
    Optional<TraceModel> findByModelCode(String modelCode);
    List<TraceModel> findByProductType(String productType);
    List<TraceModel> findByIsEnabled(Boolean isEnabled);
    List<TraceModel> findAll();
    TraceModel save(TraceModel model);
    void update(TraceModel model);
    void deleteById(Long id);
}
