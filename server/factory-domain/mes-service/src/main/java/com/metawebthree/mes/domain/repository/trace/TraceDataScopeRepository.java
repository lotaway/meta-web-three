package com.metawebthree.mes.domain.repository.trace;

import com.metawebthree.mes.domain.entity.TraceDataScope;
import java.util.List;
import java.util.Optional;

public interface TraceDataScopeRepository {
    Optional<TraceDataScope> findById(Long id);
    Optional<TraceDataScope> findByScopeCode(String scopeCode);
    List<TraceDataScope> findByScopeType(TraceDataScope.DataScopeType scopeType);
    List<TraceDataScope> findByIsDefault(Boolean isDefault);
    List<TraceDataScope> findAll();
    TraceDataScope save(TraceDataScope scope);
    void update(TraceDataScope scope);
    void deleteById(Long id);
}
