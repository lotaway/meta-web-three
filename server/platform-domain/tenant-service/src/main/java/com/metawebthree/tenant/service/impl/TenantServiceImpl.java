package com.metawebthree.tenant.service.impl;

import com.metawebthree.tenant.entity.Tenant;
import com.metawebthree.tenant.entity.TenantShop;
import com.metawebthree.tenant.entity.TenantUser;
import com.metawebthree.tenant.enums.TenantStatus;
import com.metawebthree.tenant.enums.TenantUserRole;
import com.metawebthree.tenant.mapper.TenantMapper;
import com.metawebthree.tenant.mapper.TenantShopMapper;
import com.metawebthree.tenant.mapper.TenantUserMapper;
import com.metawebthree.tenant.service.TenantService;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;

import org.apache.dubbo.config.annotation.DubboService;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@DubboService
@Service
public class TenantServiceImpl extends ServiceImpl<TenantMapper, Tenant> implements TenantService {

    private final TenantShopMapper shopMapper;
    private final TenantUserMapper tenantUserMapper;

    public TenantServiceImpl(TenantShopMapper shopMapper, TenantUserMapper tenantUserMapper) {
        this.shopMapper = shopMapper;
        this.tenantUserMapper = tenantUserMapper;
    }

    @Override
    public Tenant create(Tenant tenant) {
        tenant.setStatus(TenantStatus.PENDING.name());
        save(tenant);
        return tenant;
    }

    @Override
    public Tenant update(Tenant tenant) {
        updateById(tenant);
        return tenant;
    }

    @Override
    public Tenant getById(Long id) {
        return baseMapper.selectById(id);
    }

    @Override
    public Tenant getByCode(String code) {
        return baseMapper.selectOne(new LambdaQueryWrapper<Tenant>().eq(Tenant::getCode, code));
    }

    @Override
    public Tenant getByEmail(String email) {
        return baseMapper.selectOne(new LambdaQueryWrapper<Tenant>().eq(Tenant::getContactEmail, email));
    }

    @Override
    public IPage<Tenant> page(Page<Tenant> page, Tenant query) {
        LambdaQueryWrapper<Tenant> wrapper = new LambdaQueryWrapper<>();
        if (query != null && query.getStatus() != null) {
            wrapper.eq(Tenant::getStatus, query.getStatus());
        }
        wrapper.orderByDesc(Tenant::getCreatedAt);
        return baseMapper.selectPage(page, wrapper);
    }

    @Override
    @Transactional
    public void approve(Long id) {
        Tenant tenant = getById(id);
        if (tenant != null) {
            tenant.setStatus(TenantStatus.APPROVED.name());
            updateById(tenant);
        }
    }

    @Override
    @Transactional
    public void reject(Long id) {
        Tenant tenant = getById(id);
        if (tenant != null) {
            tenant.setStatus(TenantStatus.REJECTED.name());
            updateById(tenant);
        }
    }

    @Override
    @Transactional
    public void disable(Long id) {
        Tenant tenant = getById(id);
        if (tenant != null) {
            tenant.setStatus(TenantStatus.DISABLED.name());
            updateById(tenant);
        }
    }

    @Override
    @Transactional
    public TenantShop createShop(TenantShop shop) {
        shopMapper.insert(shop);
        return shop;
    }

    @Override
    @Transactional
    public TenantShop updateShop(TenantShop shop) {
        shopMapper.updateById(shop);
        return shop;
    }

    @Override
    public TenantShop getShopByTenant(Long tenantId) {
        return shopMapper.selectOne(new LambdaQueryWrapper<TenantShop>().eq(TenantShop::getTenantId, tenantId));
    }

    @Override
    @Transactional
    public void associateUser(Long tenantId, Long userId, TenantUserRole role) {
        TenantUser tenantUser = new TenantUser();
        tenantUser.setTenantId(tenantId);
        tenantUser.setUserId(userId);
        tenantUser.setRole(role.name());
        tenantUser.setStatus("ACTIVE");
        tenantUserMapper.insert(tenantUser);
    }

    @Override
    @Transactional
    public void removeUser(Long tenantId, Long userId) {
        tenantUserMapper.delete(new LambdaQueryWrapper<TenantUser>()
                .eq(TenantUser::getTenantId, tenantId)
                .eq(TenantUser::getUserId, userId));
    }

    @Override
    public List<TenantUser> getUsersByTenant(Long tenantId) {
        return tenantUserMapper.selectList(new LambdaQueryWrapper<TenantUser>().eq(TenantUser::getTenantId, tenantId));
    }
}
