package com.metawebthree.digitaltwin.infrastructure.persistence.converter;

import com.metawebthree.digitaltwin.domain.entity.InventoryItem;
import com.metawebthree.digitaltwin.infrastructure.persistence.dataobject.InventoryItemDO;
import org.springframework.stereotype.Component;

@Component
public class InventoryItemConverter {

    public InventoryItem toEntity(InventoryItemDO itemDO) {
        if (itemDO == null) {
            return null;
        }
        InventoryItem item = new InventoryItem();
        InventoryItemFieldAssigner.assignToEntity(item, itemDO);
        return item;
    }

    public InventoryItemDO toDO(InventoryItem item) {
        if (item == null) {
            return null;
        }
        InventoryItemDO itemDO = new InventoryItemDO();
        InventoryItemFieldAssigner.assignToDO(itemDO, item);
        return itemDO;
    }
}