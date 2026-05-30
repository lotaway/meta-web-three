package com.metawebthree.finance.domain.repository.cost;

import com.metawebthree.finance.domain.entity.cost.Activity;
import java.util.List;

public interface ActivityRepository {
    Activity save(Activity activity);
    Activity findById(Long id);
    Activity findByCode(String activityCode);
    List<Activity> findAll();
    List<Activity> findByCostCenterId(Long costCenterId);
    List<Activity> findByResourcePoolId(Long resourcePoolId);
    List<Activity> findByType(Activity.ActivityType type);
    void delete(Long id);
}