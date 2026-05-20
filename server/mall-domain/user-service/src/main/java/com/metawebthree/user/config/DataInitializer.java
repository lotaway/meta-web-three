package com.metawebthree.user.config;

import com.metawebthree.user.domain.model.*;
import com.metawebthree.user.infrastructure.config.SeedDataProperties;
import com.metawebthree.user.infrastructure.persistence.mapper.*;
import com.metawebthree.user.application.AdminService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.CommandLineRunner;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;

@Slf4j
@Component
@Order(1)
@RequiredArgsConstructor
public class DataInitializer implements CommandLineRunner {

    private final ResourceCategoryMapper resourceCategoryMapper;
    private final ResourceMapper resourceMapper;
    private final MenuMapper menuMapper;
    private final RoleMapper roleMapper;
    private final RoleResourceRelationMapper roleResourceRelationMapper;
    private final RoleMenuRelationMapper roleMenuRelationMapper;
    private final AdminService adminService;
    private final SeedDataProperties seedDataProperties;

    @Override
    @Transactional
    public void run(String... args) {
        if (roleMapper.selectCount(null) > 0) {
            log.info("RBAC seed data already exists, skipping initialization.");
            return;
        }
        log.info("Initializing RBAC seed data...");
        initResourceCategories();
        initResources();
        initMenus();
        initRoles();
        initRoleRelations();
        adminService.ensureDefaultAdmin();
        log.info("RBAC seed data initialized successfully.");
    }

    private void initResourceCategories() {
        List<ResourceCategoryDO> categories = List.of(
                category(1L, "product-management", 0),
                category(2L, "order-management", 1),
                category(3L, "user-management", 2),
                category(4L, "marketing-management", 3));
        for (ResourceCategoryDO c : categories) {
            resourceCategoryMapper.insert(c);
        }
    }

    private ResourceCategoryDO category(Long id, String name, int sort) {
        ResourceCategoryDO c = new ResourceCategoryDO();
        c.setId(id);
        c.setName(name);
        c.setSort(sort);
        c.setCreateTime(LocalDateTime.now());
        return c;
    }

    private void initResources() {
        List<ResourceDO> resources = List.of(
                // 商品管理 (categoryId=1)
                resource(101L, "view-product-list", "/product/list", "pms:product:read", 1L),
                resource(102L, "create-product", "/product/create", "pms:product:create", 1L),
                resource(103L, "update-product", "/product/update/*", "pms:product:update", 1L),
                resource(104L, "delete-product", "/product/delete/*", "pms:product:delete", 1L),
                resource(105L, "view-product-category", "/productCategory/list/*", "pms:category:read", 1L),
                resource(106L, "create-product-category", "/productCategory/create", "pms:category:create", 1L),
                resource(107L, "update-product-category", "/productCategory/update/*", "pms:category:update", 1L),
                resource(108L, "delete-product-category", "/productCategory/delete/*", "pms:category:delete", 1L),
                resource(109L, "view-brand", "/brand/listAll", "pms:brand:read", 1L),
                resource(110L, "create-brand", "/brand/create", "pms:brand:create", 1L),
                resource(111L, "update-brand", "/brand/update/*", "pms:brand:update", 1L),
                resource(112L, "delete-brand", "/brand/delete/*", "pms:brand:delete", 1L),
                // 订单管理 (categoryId=2)
                resource(201L, "view-order", "/order/list", "oms:order:read", 2L),
                resource(202L, "update-order", "/order/update/*", "oms:order:update", 2L),
                resource(203L, "delete-order", "/order/delete/*", "oms:order:delete", 2L),
                resource(204L, "view-return-application", "/returnApply/list", "oms:return:read", 2L),
                resource(205L, "review-return", "/returnApply/update/*", "oms:return:update", 2L),
                resource(206L, "delete-return-application", "/returnApply/delete/*", "oms:return:delete", 2L),
                // 用户管理 (categoryId=3)
                resource(301L, "view-admin-list", "/admin/list", "ums:admin:read", 3L),
                resource(302L, "create-admin", "/admin/register", "ums:admin:create", 3L),
                resource(303L, "update-admin", "/admin/update/*", "ums:admin:update", 3L),
                resource(304L, "delete-admin", "/admin/delete/*", "ums:admin:delete", 3L),
                resource(305L, "view-user-list", "/user/list", "ums:user:read", 3L),
                resource(306L, "update-user-status", "/user/updateStatus/*", "ums:user:update", 3L),
                resource(307L, "view-role-list", "/role/list", "ums:role:read", 3L),
                resource(308L, "create-role", "/role/create", "ums:role:create", 3L),
                resource(309L, "update-role", "/role/update/*", "ums:role:update", 3L),
                resource(310L, "delete-role", "/role/delete/*", "ums:role:delete", 3L),
                // 营销管理 (categoryId=4)
                resource(401L, "view-coupon", "/coupon/list", "sms:coupon:read", 4L),
                resource(402L, "create-coupon", "/coupon/create", "sms:coupon:create", 4L),
                resource(403L, "update-coupon", "/coupon/update/*", "sms:coupon:update", 4L),
                resource(404L, "delete-coupon", "/coupon/delete/*", "sms:coupon:delete", 4L),
                resource(405L, "view-flash-sale", "/flash/list", "sms:flash:read", 4L),
                resource(406L, "create-flash-sale", "/flash/create", "sms:flash:create", 4L),
                resource(407L, "update-flash-sale", "/flash/update/*", "sms:flash:update", 4L),
                resource(408L, "delete-flash-sale", "/flash/delete/*", "sms:flash:delete", 4L),
                resource(409L, "view-flash-session", "/flashSession/list", "sms:flashSession:read", 4L),
                resource(410L, "update-flash-session", "/flashSession/update/*", "sms:flashSession:update", 4L),
                // 菜单管理 (categoryId=3)
                resource(311L, "view-menu", "/menu/treeList", "ums:menu:read", 3L),
                resource(312L, "create-menu", "/menu/create", "ums:menu:create", 3L),
                resource(313L, "update-menu", "/menu/update/*", "ums:menu:update", 3L),
                resource(314L, "delete-menu", "/menu/delete/*", "ums:menu:delete", 3L),
                // 资源管理 (categoryId=3)
                resource(315L, "view-resource", "/resource/list", "ums:resource:read", 3L),
                resource(316L, "create-resource", "/resource/create", "ums:resource:create", 3L),
                resource(317L, "update-resource", "/resource/update/*", "ums:resource:update", 3L),
                resource(318L, "delete-resource", "/resource/delete/*", "ums:resource:delete", 3L),
                // 资源分类管理 (categoryId=3)
                resource(319L, "view-resource-category", "/resourceCategory/listAll", "ums:resourceCategory:read", 3L),
                resource(320L, "create-resource-category", "/resourceCategory/create", "ums:resourceCategory:create", 3L),
                resource(321L, "update-resource-category", "/resourceCategory/update/*", "ums:resourceCategory:update", 3L),
                resource(322L, "delete-resource-category", "/resourceCategory/delete/*", "ums:resourceCategory:delete", 3L));
        for (ResourceDO r : resources) {
            resourceMapper.insert(r);
        }
    }

    private ResourceDO resource(Long id, String name, String url, String value, Long categoryId) {
        ResourceDO r = new ResourceDO();
        r.setId(id);
        r.setName(name);
        r.setUrl(url);
        r.setValue(value);
        r.setCategoryId(categoryId);
        r.setCreateTime(LocalDateTime.now());
        return r;
    }

    private void initMenus() {
        LocalDateTime now = LocalDateTime.now();
        // Level 1 menus
        menuMapper.insert(menu(2001L, 0L, now, "product-management", 0, 0, "pms", "product", 0));
        menuMapper.insert(menu(2002L, 0L, now, "order-management", 0, 1, "oms", "order", 0));
        menuMapper.insert(menu(2003L, 0L, now, "user-management", 0, 2, "ums", "user", 0));
        menuMapper.insert(menu(2004L, 0L, now, "marketing-management", 0, 3, "sms", "sms", 0));
        // Level 2 menus - product-management
        menuMapper.insert(menu(2011L, 2001L, now, "product-list", 1, 0, "pmsProduct", "product", 0));
        menuMapper.insert(menu(2012L, 2001L, now, "product-category", 1, 1, "pmsProductCategory", "product-category", 0));
        menuMapper.insert(menu(2013L, 2001L, now, "brand-management", 1, 2, "pmsBrand", "brand", 0));
        // Level 2 menus - order-management
        menuMapper.insert(menu(2021L, 2002L, now, "order-list", 1, 0, "omsOrder", "order", 0));
        menuMapper.insert(menu(2022L, 2002L, now, "return-application", 1, 1, "omsOrderReturn", "return", 0));
        // Level 2 menus - user-management
        menuMapper.insert(menu(2031L, 2003L, now, "user-list", 1, 0, "umsUser", "user", 0));
        menuMapper.insert(menu(2032L, 2003L, now, "admin-list", 1, 1, "umsAdmin", "admin", 0));
        menuMapper.insert(menu(2033L, 2003L, now, "role-management", 1, 2, "umsRole", "role", 0));
        // Level 2 menus - marketing-management
        menuMapper.insert(menu(2041L, 2004L, now, "coupon-management", 1, 0, "smsCoupon", "coupon", 0));
        menuMapper.insert(menu(2042L, 2004L, now, "flash-sale", 1, 1, "smsFlashPromotion", "flash", 0));
    }

    private MenuDO menu(Long id, Long parentId, LocalDateTime createTime, String title, int level, int sort, String name, String icon, int hidden) {
        MenuDO m = new MenuDO();
        m.setId(id);
        m.setParentId(parentId);
        m.setCreateTime(createTime);
        m.setTitle(title);
        m.setLevel(level);
        m.setSort(sort);
        m.setName(name);
        m.setIcon(icon);
        m.setHidden(hidden);
        return m;
    }

    private void initRoles() {
        LocalDateTime now = LocalDateTime.now();
        // super-admin
        RoleDO superAdmin = new RoleDO();
        superAdmin.setId(3001L);
        superAdmin.setName("super-admin");
        superAdmin.setDescription("system-owner-with-all-permissions");
        superAdmin.setAdminCount(0);
        superAdmin.setCreateTime(now);
        superAdmin.setStatus(1);
        superAdmin.setSort(0);
        roleMapper.insert(superAdmin);
        // operations-specialist
        RoleDO ops = new RoleDO();
        ops.setId(3002L);
        ops.setName("operations-specialist");
        ops.setDescription("daily-operations-product-management-marketing");
        ops.setAdminCount(0);
        ops.setCreateTime(now);
        ops.setStatus(1);
        ops.setSort(1);
        roleMapper.insert(ops);
        // customer-service
        RoleDO cs = new RoleDO();
        cs.setId(3003L);
        cs.setName("customer-service");
        cs.setDescription("after-sales-order-viewing-user-viewing");
        cs.setAdminCount(0);
        cs.setCreateTime(now);
        cs.setStatus(1);
        cs.setSort(2);
        roleMapper.insert(cs);
        // warehouse-clerk
        RoleDO wh = new RoleDO();
        wh.setId(3004L);
        wh.setName("warehouse-clerk");
        wh.setDescription("shipping-management");
        wh.setAdminCount(0);
        wh.setCreateTime(now);
        wh.setStatus(1);
        wh.setSort(3);
        roleMapper.insert(wh);
    }

    private void initRoleRelations() {
        long superAdminRoleId = seedDataProperties.getSuperAdminRoleId();
        // super-admin: all resources + all menus
        for (long resid = seedDataProperties.getResource().getStartId();
             resid <= seedDataProperties.getResource().getEndId(); resid++) {
            if (resid == 300L || resid == 400L) continue;
            roleResourceRelationMapper.insert(relation(superAdminRoleId, resid));
        }
        for (long menuid = seedDataProperties.getMenu().getStartId();
             menuid <= seedDataProperties.getMenu().getEndId(); menuid++) {
            roleMenuRelationMapper.insert(menuRelation(superAdminRoleId, menuid));
        }
        // operations-specialist (3002): pms:* + sms:* resources + product/marketing menus
        long[] opsResources = {101L, 102L, 103L, 104L, 105L, 106L, 107L, 108L, 109L, 110L, 111L, 112L,
                               401L, 402L, 403L, 404L, 405L, 406L, 407L, 408L, 409L, 410L};
        for (long resid : opsResources) {
            roleResourceRelationMapper.insert(relation(3002L, resid));
        }
        roleMenuRelationMapper.insert(menuRelation(3002L, 2001L));
        roleMenuRelationMapper.insert(menuRelation(3002L, 2011L));
        roleMenuRelationMapper.insert(menuRelation(3002L, 2012L));
        roleMenuRelationMapper.insert(menuRelation(3002L, 2013L));
        roleMenuRelationMapper.insert(menuRelation(3002L, 2004L));
        roleMenuRelationMapper.insert(menuRelation(3002L, 2041L));
        roleMenuRelationMapper.insert(menuRelation(3002L, 2042L));
        // customer-service (3003): oms:read, oms:return:*, ums:user:read resources + order/user menus
        long[] csResources = {201L, 204L, 205L, 206L, 305L};
        for (long resid : csResources) {
            roleResourceRelationMapper.insert(relation(3003L, resid));
        }
        roleMenuRelationMapper.insert(menuRelation(3003L, 2002L));
        roleMenuRelationMapper.insert(menuRelation(3003L, 2021L));
        roleMenuRelationMapper.insert(menuRelation(3003L, 2022L));
        roleMenuRelationMapper.insert(menuRelation(3003L, 2003L));
        roleMenuRelationMapper.insert(menuRelation(3003L, 2031L));
        // warehouse-clerk (3004): oms:order:read/update + order menus
        long[] whResources = {201L, 202L};
        for (long resid : whResources) {
            roleResourceRelationMapper.insert(relation(3004L, resid));
        }
        roleMenuRelationMapper.insert(menuRelation(3004L, 2002L));
        roleMenuRelationMapper.insert(menuRelation(3004L, 2021L));
    }

    private RoleResourceRelationDO relation(Long roleId, Long resourceId) {
        RoleResourceRelationDO r = new RoleResourceRelationDO();
        r.setRoleId(roleId);
        r.setResourceId(resourceId);
        return r;
    }

    private RoleMenuRelationDO menuRelation(Long roleId, Long menuId) {
        RoleMenuRelationDO r = new RoleMenuRelationDO();
        r.setRoleId(roleId);
        r.setMenuId(menuId);
        return r;
    }

}
