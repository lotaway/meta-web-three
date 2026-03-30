package com.metawebthree.order.application;

import com.metawebthree.order.domain.model.OrderSetting;
import com.metawebthree.order.domain.repository.OrderSettingRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class OrderSettingApplicationService {

    private final OrderSettingRepository repository;

    public void defineSetting(OrderSetting setting) {
        validateSetting(setting);
        repository.save(setting);
    }

    public void updateSetting(OrderSetting setting) {
        repository.update(setting);
    }

    public OrderSetting getSetting() {
        return repository.findDefault();
    }

    private void validateSetting(OrderSetting setting) {
        if (setting == null) {
            throw new IllegalArgumentException("Order setting mapping is required");
        }
    }
}
