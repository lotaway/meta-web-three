package com.metawebthree.digitaltwin.infrastructure.persistence.converter;

import com.metawebthree.digitaltwin.domain.entity.Shelf;
import com.metawebthree.digitaltwin.infrastructure.persistence.dataobject.ShelfDO;
import org.springframework.stereotype.Component;

@Component
public class ShelfConverter {

    public Shelf toEntity(ShelfDO shelfDO) {
        if (shelfDO == null) {
            return null;
        }
        Shelf shelf = new Shelf();
        ShelfFieldAssigner.assignToEntity(shelf, shelfDO);
        return shelf;
    }

    public ShelfDO toDO(Shelf shelf) {
        if (shelf == null) {
            return null;
        }
        ShelfDO shelfDO = new ShelfDO();
        ShelfFieldAssigner.assignToDO(shelfDO, shelf);
        return shelfDO;
    }
}