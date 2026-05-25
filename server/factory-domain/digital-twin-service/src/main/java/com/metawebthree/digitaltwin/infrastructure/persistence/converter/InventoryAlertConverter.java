package com.metawebthree.digitaltwin.infrastructure.persistence.converter;

import com.metawebthree.digitaltwin.domain.entity.InventoryAlert;
import com.metawebthree.digitaltwin.infrastructure.persistence.dataobject.InventoryAlertDO;
import org.springframework.stereotype.Component;

@Component
public class InventoryAlertConverter {

    public InventoryAlert toEntity(InventoryAlertDO alertDO) {
        if (alertDO == null) {
            return null;
        }
        InventoryAlert alert = new InventoryAlert();
        InventoryAlertFieldAssigner.assignToEntity(alert, alertDO);
        return alert;
    }

    public InventoryAlertDO toDO(InventoryAlert alert) {
        if (alert == null) {
            return null;
        }
        InventoryAlertDO alertDO = new InventoryAlertDO();
        InventoryAlertFieldAssigner.assignToDO(alertDO, alert);
        return alertDO;
    }
}