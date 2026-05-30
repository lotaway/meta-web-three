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
    
    // Supplier permissions
    public static final String SUPPLIER_READ     = "sc:supplier:read";
    public static final String SUPPLIER_CREATE   = "sc:supplier:create";
    public static final String SUPPLIER_UPDATE   = "sc:supplier:update";
    public static final String SUPPLIER_DELETE   = "sc:supplier:delete";
    public static final String SUPPLIER_ASSESS   = "sc:supplier:assess";
    
    // Procurement permissions
    public static final String PROCUREMENT_READ    = "sc:procurement:read";
    public static final String PROCUREMENT_CREATE  = "sc:procurement:create";
    public static final String PROCUREMENT_UPDATE  = "sc:procurement:update";
    public static final String PROCUREMENT_APPROVE = "sc:procurement:approve";
    public static final String PROCUREMENT_REJECT  = "sc:procurement:reject";

    // ABC Classification permissions
    public static final String INVENTORY_ABC_ANALYZE = "sc:inventory:abc:analyze";
    
    // Outbound Strategy permissions
    public static final String OUTBOUND_STRATEGY_READ = "sc:outbound:strategy:read";
    public static final String OUTBOUND_STRATEGY_CREATE = "sc:outbound:strategy:create";
    public static final String OUTBOUND_STRATEGY_UPDATE = "sc:outbound:strategy:update";
    public static final String OUTBOUND_STRATEGY_DELETE = "sc:outbound:strategy:delete";
    public static final String OUTBOUND_ALLOCATE = "sc:outbound:allocate";
    
    // Supplier Portal permissions (供应商协同门户)
    public static final String PORTAL_ORDER_READ = "sc:portal:order:read";
    public static final String PORTAL_SHIPMENT_READ = "sc:portal:shipment:read";
    public static final String PORTAL_SHIPMENT_CREATE = "sc:portal:shipment:create";
    public static final String PORTAL_SHIPMENT_UPDATE = "sc:portal:shipment:update";
    public static final String PORTAL_SHIPMENT_SUBMIT = "sc:portal:shipment:submit";
    public static final String PORTAL_SHIPMENT_CONFIRM = "sc:portal:shipment:confirm";
    public static final String PORTAL_RECONCILIATION_READ = "sc:portal:reconciliation:read";
    public static final String PORTAL_RECONCILIATION_CREATE = "sc:portal:reconciliation:create";
    public static final String PORTAL_RECONCILIATION_SUBMIT = "sc:portal:reconciliation:submit";
    public static final String PORTAL_RECONCILIATION_CONFIRM = "sc:portal:reconciliation:confirm";
    public static final String PORTAL_RECONCILIATION_REJECT = "sc:portal:reconciliation:reject";
    public static final String PORTAL_RECONCILIATION_PAID = "sc:portal:reconciliation:paid";
}
