package com.metawebthree.inventory.common;

/**
 * 供应链权限定义 - 库存服务
 */
public final class SupplyChainPermissions {

    private SupplyChainPermissions() {}

    // ==================== 库存服务权限 ====================
    public static final String INVENTORY_READ     = "sc:inventory:read";
    public static final String INVENTORY_RESERVE  = "sc:inventory:reserve";
    public static final String INVENTORY_CONFIRM = "sc:inventory:confirm";
    public static final String INVENTORY_CANCEL   = "sc:inventory:cancel";
    public static final String INVENTORY_ADJUST   = "sc:inventory:adjust";
}