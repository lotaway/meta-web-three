package com.metawebthree.finance.infrastructure.persistence.repository.cost;

import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.metawebthree.finance.domain.entity.cost.Activity;
import com.metawebthree.finance.domain.repository.cost.ActivityRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * 作业成本仓储（本地开发用内存实现）。
 */
@Repository
public class ActivityRepositoryImpl implements ActivityRepository {

    private final Map<Long, Activity> store = new ConcurrentHashMap<>();

    @Override
    public Activity save(Activity activity) {
        if (activity.getId() == null) {
            activity.setId(IdWorker.getId());
        }
        store.put(activity.getId(), activity);
        return activity;
    }

    @Override
    public Activity findById(Long id) {
        return store.get(id);
    }

    @Override
    public Activity findByCode(String activityCode) {
        return store.values().stream()
                .filter(a -> Objects.equals(a.getActivityCode(), activityCode))
                .findFirst()
                .orElse(null);
    }

    @Override
    public List<Activity> findAll() {
        return store.values().stream().collect(Collectors.toList());
    }

    @Override
    public List<Activity> findByCostCenterId(Long costCenterId) {
        return store.values().stream()
                .filter(a -> Objects.equals(a.getCostCenterId(), costCenterId))
                .collect(Collectors.toList());
    }

    @Override
    public List<Activity> findByResourcePoolId(Long resourcePoolId) {
        return store.values().stream()
                .filter(a -> Objects.equals(a.getResourcePoolId(), resourcePoolId))
                .collect(Collectors.toList());
    }

    @Override
    public List<Activity> findByType(Activity.ActivityType type) {
        return store.values().stream()
                .filter(a -> Objects.equals(a.getType(), type))
                .collect(Collectors.toList());
    }

    @Override
    public void delete(Long id) {
        store.remove(id);
    }
}
