package com.metawebthree.tenant.service.impl;

import com.metawebthree.tenant.entity.Tenant;
import com.metawebthree.tenant.entity.TenantShop;
import com.metawebthree.tenant.entity.TenantUser;
import com.metawebthree.tenant.enums.TenantStatus;
import com.metawebthree.tenant.enums.TenantUserRole;
import com.metawebthree.tenant.mapper.TenantMapper;
import com.metawebthree.tenant.mapper.TenantShopMapper;
import com.metawebthree.tenant.mapper.TenantUserMapper;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.test.util.ReflectionTestUtils;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class TenantServiceImplTest {

    @Mock
    private TenantMapper baseMapper;

    @Mock
    private TenantShopMapper shopMapper;

    @Mock
    private TenantUserMapper tenantUserMapper;

    private TenantServiceImpl tenantService;

    private Tenant testTenant;

    @BeforeEach
    void setUp() {
        tenantService = new TenantServiceImpl(shopMapper, tenantUserMapper);
        ReflectionTestUtils.setField(tenantService, "baseMapper", baseMapper);
        testTenant = new Tenant();
        testTenant.setId(1L);
        testTenant.setName("Test Tenant");
        testTenant.setCode("TENANT_001");
        testTenant.setContactEmail("admin@example.com");
    }

    @Test
    void testCreate_SetsPendingStatus() {
        // When
        Tenant result = tenantService.create(testTenant);

        // Then
        assertEquals(TenantStatus.PENDING.name(), result.getStatus());
        verify(baseMapper).insert(any(Tenant.class));
    }

    @Test
    void testGetById() {
        // Given
        when(baseMapper.selectById(1L)).thenReturn(testTenant);

        // When
        Tenant result = tenantService.getById(1L);

        // Then
        assertEquals("TENANT_001", result.getCode());
    }

    @Test
    void testApprove_UpdatesStatus() {
        // Given
        when(baseMapper.selectById(1L)).thenReturn(testTenant);

        // When
        tenantService.approve(1L);

        // Then
        assertEquals(TenantStatus.APPROVED.name(), testTenant.getStatus());
        verify(baseMapper).updateById(testTenant);
    }

    @Test
    void testReject_UpdatesStatus() {
        // Given
        when(baseMapper.selectById(1L)).thenReturn(testTenant);

        // When
        tenantService.reject(1L);

        // Then
        assertEquals(TenantStatus.REJECTED.name(), testTenant.getStatus());
        verify(baseMapper).updateById(testTenant);
    }

    @Test
    void testDisable_UpdatesStatus() {
        // Given
        when(baseMapper.selectById(1L)).thenReturn(testTenant);

        // When
        tenantService.disable(1L);

        // Then
        assertEquals(TenantStatus.DISABLED.name(), testTenant.getStatus());
        verify(baseMapper).updateById(testTenant);
    }

    @Test
    void testCreateShop() {
        // Given
        TenantShop shop = new TenantShop();
        shop.setTenantId(1L);

        // When
        TenantShop result = tenantService.createShop(shop);

        // Then
        assertEquals(1L, result.getTenantId());
        verify(shopMapper).insert(shop);
    }

    @Test
    void testAssociateUser_InsertsActiveRelation() {
        // When
        tenantService.associateUser(1L, 100L, TenantUserRole.OPERATOR);

        // Then
        verify(tenantUserMapper).insert(any(TenantUser.class));
    }

    @Test
    void testRemoveUser() {
        // When
        tenantService.removeUser(1L, 100L);

        // Then
        verify(tenantUserMapper).delete(any());
    }
}
