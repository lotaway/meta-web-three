package com.metawebthree.routeoptimizer.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.metawebthree.routeoptimizer.domain.entity.RoutePlan;
import com.metawebthree.routeoptimizer.domain.entity.RoutePoint;
import com.metawebthree.routeoptimizer.domain.repository.RoutePlanRepository;
import com.metawebthree.routeoptimizer.infrastructure.persistence.mapper.RoutePlanMapper;
import com.metawebthree.routeoptimizer.infrastructure.persistence.mapper.RoutePointMapper;
import org.springframework.stereotype.Repository;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Repository
public class RoutePlanRepositoryImpl implements RoutePlanRepository {

    private final RoutePlanMapper planMapper;
    private final RoutePointMapper pointMapper;

    public RoutePlanRepositoryImpl(RoutePlanMapper planMapper, RoutePointMapper pointMapper) {
        this.planMapper = planMapper;
        this.pointMapper = pointMapper;
    }

    @Override
    public Optional<RoutePlan> findById(Long id) {
        RoutePlan plan = planMapper.selectById(id);
        if (plan != null) {
            plan.setPoints(pointMapper.selectList(
                new QueryWrapper<RoutePoint>().eq("route_plan_id", id).orderByAsc("sequence")));
        }
        return Optional.ofNullable(plan);
    }

    @Override
    public Optional<RoutePlan> findByPlanCode(String planCode) {
        RoutePlan plan = planMapper.selectOne(
            new QueryWrapper<RoutePlan>().eq("plan_code", planCode));
        if (plan != null) {
            plan.setPoints(pointMapper.selectList(
                new QueryWrapper<RoutePoint>().eq("route_plan_id", plan.getId()).orderByAsc("sequence")));
        }
        return Optional.ofNullable(plan);
    }

    @Override
    public List<RoutePlan> findByStatus(RoutePlan.RouteStatus status) {
        List<RoutePlan> plans = planMapper.selectList(
            new QueryWrapper<RoutePlan>().eq("status", status));
        for (RoutePlan plan : plans) {
            plan.setPoints(pointMapper.selectList(
                new QueryWrapper<RoutePoint>().eq("route_plan_id", plan.getId()).orderByAsc("sequence")));
        }
        return plans;
    }

    @Override
    public List<RoutePlan> findByVehicleCode(String vehicleCode) {
        List<RoutePlan> plans = planMapper.selectList(
            new QueryWrapper<RoutePlan>().eq("vehicle_code", vehicleCode));
        for (RoutePlan plan : plans) {
            plan.setPoints(pointMapper.selectList(
                new QueryWrapper<RoutePoint>().eq("route_plan_id", plan.getId()).orderByAsc("sequence")));
        }
        return plans;
    }

    @Override
    public List<RoutePlan> findAll() {
        List<RoutePlan> plans = planMapper.selectList(null);
        for (RoutePlan plan : plans) {
            plan.setPoints(pointMapper.selectList(
                new QueryWrapper<RoutePoint>().eq("route_plan_id", plan.getId()).orderByAsc("sequence")));
        }
        return plans;
    }

    @Override
    public void save(RoutePlan routePlan) {
        if (routePlan.getId() != null) {
            planMapper.updateById(routePlan);
            pointMapper.delete(new QueryWrapper<RoutePoint>().eq("route_plan_id", routePlan.getId()));
        } else {
            planMapper.insert(routePlan);
        }
        if (routePlan.getPoints() != null) {
            for (RoutePoint point : routePlan.getPoints()) {
                point.setRoutePlanId(routePlan.getId());
                pointMapper.insert(point);
            }
        }
    }

    @Override
    public void delete(RoutePlan routePlan) {
        if (routePlan.getId() != null) {
            pointMapper.delete(new QueryWrapper<RoutePoint>().eq("route_plan_id", routePlan.getId()));
            planMapper.deleteById(routePlan.getId());
        }
    }

    @Override
    public List<RoutePlan> findByPlannedStartTimeBetween(LocalDateTime start, LocalDateTime end) {
        List<RoutePlan> plans = planMapper.selectList(
            new QueryWrapper<RoutePlan>().between("planned_start_time", start, end));
        for (RoutePlan plan : plans) {
            plan.setPoints(pointMapper.selectList(
                new QueryWrapper<RoutePoint>().eq("route_plan_id", plan.getId()).orderByAsc("sequence")));
        }
        return plans;
    }
}
