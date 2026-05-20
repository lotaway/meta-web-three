package com.metawebthree.inventory.domain.event;

public enum InventoryEventType {
    RESERVED("inventory_reserved"),
    CONFIRMED("inventory_confirmed"),
    CANCELLED("inventory_cancelled"),
    INCREASED("inventory_increased"),
    DECREASED("inventory_decreased"),
    LOW_STOCK("inventory_low_stock");

    private final String code;

    InventoryEventType(String code) {
        this.code = code;
    }

    public String getCode() {
        return code;
    }
}