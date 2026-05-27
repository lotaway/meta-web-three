package com.metawebthree.common;

public final class SupplyChainPermissions {

    private SupplyChainPermissions() {}

    public static final String WAREHOUSE_READ    = "sc:warehouse:read";
    public static final String WAREHOUSE_CREATE  = "sc:warehouse:create";
    public static final String WAREHOUSE_UPDATE  = "sc:warehouse:update";
    public static final String WAREHOUSE_DELETE  = "sc:warehouse:delete";
    
    public static final String INBOUND_READ      = "sc:inbound:read";
    public static final String INBOUND_CREATE    = "sc:inbound:create";
    public static final String INBOUND_CONFIRM   = "sc:inbound:confirm";
    public static final String INBOUND_COMPLETE  = "sc:inbound:complete";
    
    public static final String OUTBOUND_READ     = "sc:outbound:read";
    public static final String OUTBOUND_CREATE   = "sc:outbound:create";
    public static final String OUTBOUND_CONFIRM  = "sc:outbound:confirm";
    public static final String OUTBOUND_COMPLETE = "sc:outbound:complete";

    public static final String INVENTORY_READ     = "sc:inventory:read";
    public static final String INVENTORY_RESERVE  = "sc:inventory:reserve";
    public static final String INVENTORY_CONFIRM = "sc:inventory:confirm";
    public static final String INVENTORY_CANCEL   = "sc:inventory:cancel";
    public static final String INVENTORY_ADJUST   = "sc:inventory:adjust";
}
