package com.metawebthree.tenant.service;

import com.metawebthree.tenant.entity.Tenant;
import com.metawebthree.tenant.entity.TenantShop;
import com.metawebthree.tenant.entity.TenantUser;
import com.metawebthree.tenant.enums.TenantUserRole;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;

import java.util.List;

public interface TenantService {
    Tenant create(Tenant tenant);
    Tenant update(Tenant tenant);
    Tenant getById(Long id);
    Tenant getByCode(String code);
    Tenant getByEmail(String email);
    IPage<Tenant> page(Page<Tenant> page, Tenant query);

    void approve(Long id);
    void reject(Long id);
    void disable(Long id);

    TenantShop createShop(TenantShop shop);
    TenantShop updateShop(TenantShop shop);
    TenantShop getShopByTenant(Long tenantId);

    void associateUser(Long tenantId, Long userId, TenantUserRole role);
    void removeUser(Long tenantId, Long userId);
    List<TenantUser> getUsersByTenant(Long tenantId);
}
