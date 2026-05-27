package com.metawebthree.warehouse.common;

/**
 * 供应链权限定义 - 仓库服务
 */
public final class SupplyChainPermissions {

    private SupplyChainPermissions() {}

    // ==================== 仓库服务权限 ====================
    // 仓库管理
    public static final String WAREHOUSE_READ    = "sc:warehouse:read";
    public static final String WAREHOUSE_CREATE  = "sc:warehouse:create";
    public static final String WAREHOUSE_UPDATE  = "sc:warehouse:update";
    public static final String WAREHOUSE_DELETE  = "sc:warehouse:delete";
    
    // 入库管理
    public static final String INBOUND_READ      = "sc:inbound:read";
    public static final String INBOUND_CREATE    = "sc:inbound:create";
    public static final String INBOUND_CONFIRM   = "sc:inbound:confirm";
    public static final String INBOUND_COMPLETE  = "sc:inbound:complete";
    
    // 出库管理
    public static final String OUTBOUND_READ     = "sc:outbound:read";
    public static final String OUTBOUND_CREATE   = "sc:outbound:create";
    public static final String OUTBOUND_CONFIRM  = "sc:outbound:confirm";
    public static final String OUTBOUND_COMPLETE = "sc:outbound:complete";
}