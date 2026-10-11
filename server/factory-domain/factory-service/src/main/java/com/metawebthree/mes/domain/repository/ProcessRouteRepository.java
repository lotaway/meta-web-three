package com.metawebthree.mes.domain.repository;

import com.metawebthree.mes.domain.entity.ProcessRoute;
import java.util.List;
import java.util.Optional;

public interface ProcessRouteRepository {
    Optional<ProcessRoute> findById(Long id);
    Optional<ProcessRoute> findByRouteCode(String routeCode);
    List<ProcessRoute> findByProductCode(String productCode);
    List<ProcessRoute> findByStatus(ProcessRoute.RouteStatus status);
    ProcessRoute save(ProcessRoute route);
    void update(ProcessRoute route);
    void deleteById(Long id);
}