package com.metawebthree.user.config;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.user.domain.model.*;
import com.metawebthree.user.infrastructure.persistence.mapper.*;
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
    private final AdminMapper adminMapper;
    private final AdminRoleRelationMapper adminRoleRelationMapper;

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
        initDefaultAdmin();
        log.info("RBAC seed data initialized successfully.");
    }

    private void initResourceCategories() {
        List<ResourceCategoryDO> categories = List.of(
                category(1L, "商品管理", 0),
                category(2L, "订单管理", 1),
                category(3L, "用户管理", 2),
                category(4L, "营销管理", 3));
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
                resource(101L, "查看商品列表", "/product/list", "pms:product:read", 1L),
                resource(102L, "新增商品", "/product/create", "pms:product:create", 1L),
                resource(103L, "编辑商品", "/product/update/*", "pms:product:update", 1L),
                resource(104L, "删除商品", "/product/delete/*", "pms:product:delete", 1L),
                resource(105L, "查看商品分类", "/productCategory/list/*", "pms:category:read", 1L),
                resource(106L, "新增商品分类", "/productCategory/create", "pms:category:create", 1L),
                resource(107L, "编辑商品分类", "/productCategory/update/*", "pms:category:update", 1L),
                resource(108L, "删除商品分类", "/productCategory/delete/*", "pms:category:delete", 1L),
                resource(109L, "查看品牌", "/brand/listAll", "pms:brand:read", 1L),
                resource(110L, "新增品牌", "/brand/create", "pms:brand:create", 1L),
                resource(111L, "编辑品牌", "/brand/update/*", "pms:brand:update", 1L),
                resource(112L, "删除品牌", "/brand/delete/*", "pms:brand:delete", 1L),
                // 订单管理 (categoryId=2)
                resource(201L, "查看订单", "/order/list", "oms:order:read", 2L),
                resource(202L, "编辑订单", "/order/update/*", "oms:order:update", 2L),
                resource(203L, "删除订单", "/order/delete/*", "oms:order:delete", 2L),
                resource(204L, "查看退货申请", "/returnApply/list", "oms:return:read", 2L),
                resource(205L, "审核退货", "/returnApply/update/*", "oms:return:update", 2L),
                resource(206L, "删除退货申请", "/returnApply/delete/*", "oms:return:delete", 2L),
                // 用户管理 (categoryId=3)
                resource(301L, "查看管理员列表", "/admin/list", "ums:admin:read", 3L),
                resource(302L, "新增管理员", "/admin/register", "ums:admin:create", 3L),
                resource(303L, "修改管理员", "/admin/update/*", "ums:admin:update", 3L),
                resource(304L, "删除管理员", "/admin/delete/*", "ums:admin:delete", 3L),
                resource(305L, "查看用户列表", "/user/list", "ums:user:read", 3L),
                resource(306L, "修改用户状态", "/user/updateStatus/*", "ums:user:update", 3L),
                resource(307L, "查看角色列表", "/role/list", "ums:role:read", 3L),
                resource(308L, "新增角色", "/role/create", "ums:role:create", 3L),
                resource(309L, "修改角色", "/role/update/*", "ums:role:update", 3L),
                resource(310L, "删除角色", "/role/delete/*", "ums:role:delete", 3L),
                // 营销管理 (categoryId=4)
                resource(401L, "查看优惠券", "/coupon/list", "sms:coupon:read", 4L),
                resource(402L, "新增优惠券", "/coupon/create", "sms:coupon:create", 4L),
                resource(403L, "编辑优惠券", "/coupon/update/*", "sms:coupon:update", 4L),
                resource(404L, "删除优惠券", "/coupon/delete/*", "sms:coupon:delete", 4L),
                resource(405L, "查看秒杀活动", "/flash/list", "sms:flash:read", 4L),
                resource(406L, "新增秒杀活动", "/flash/create", "sms:flash:create", 4L),
                resource(407L, "编辑秒杀活动", "/flash/update/*", "sms:flash:update", 4L),
                resource(408L, "删除秒杀活动", "/flash/delete/*", "sms:flash:delete", 4L),
                resource(409L, "查看秒杀场次", "/flashSession/list", "sms:flashSession:read", 4L),
                resource(410L, "编辑秒杀场次", "/flashSession/update/*", "sms:flashSession:update", 4L),
                // 菜单管理 (categoryId=3)
                resource(311L, "查看菜单", "/menu/treeList", "ums:menu:read", 3L),
                resource(312L, "新增菜单", "/menu/create", "ums:menu:create", 3L),
                resource(313L, "修改菜单", "/menu/update/*", "ums:menu:update", 3L),
                resource(314L, "删除菜单", "/menu/delete/*", "ums:menu:delete", 3L),
                // 资源管理 (categoryId=3)
                resource(315L, "查看资源", "/resource/list", "ums:resource:read", 3L),
                resource(316L, "新增资源", "/resource/create", "ums:resource:create", 3L),
                resource(317L, "修改资源", "/resource/update/*", "ums:resource:update", 3L),
                resource(318L, "删除资源", "/resource/delete/*", "ums:resource:delete", 3L),
                // 资源分类管理 (categoryId=3)
                resource(319L, "查看资源分类", "/resourceCategory/listAll", "ums:resourceCategory:read", 3L),
                resource(320L, "新增资源分类", "/resourceCategory/create", "ums:resourceCategory:create", 3L),
                resource(321L, "修改资源分类", "/resourceCategory/update/*", "ums:resourceCategory:update", 3L),
                resource(322L, "删除资源分类", "/resourceCategory/delete/*", "ums:resourceCategory:delete", 3L));
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
        menuMapper.insert(menu(2001L, 0L, now, "商品管理", 0, 0, "pms", "product", 0));
        menuMapper.insert(menu(2002L, 0L, now, "订单管理", 0, 1, "oms", "order", 0));
        menuMapper.insert(menu(2003L, 0L, now, "用户管理", 0, 2, "ums", "user", 0));
        menuMapper.insert(menu(2004L, 0L, now, "营销管理", 0, 3, "sms", "sms", 0));
        // Level 2 menus - 商品管理
        menuMapper.insert(menu(2011L, 2001L, now, "商品列表", 1, 0, "pmsProduct", "product", 0));
        menuMapper.insert(menu(2012L, 2001L, now, "商品分类", 1, 1, "pmsProductCategory", "product-category", 0));
        menuMapper.insert(menu(2013L, 2001L, now, "品牌管理", 1, 2, "pmsBrand", "brand", 0));
        // Level 2 menus - 订单管理
        menuMapper.insert(menu(2021L, 2002L, now, "订单列表", 1, 0, "omsOrder", "order", 0));
        menuMapper.insert(menu(2022L, 2002L, now, "退货申请处理", 1, 1, "omsOrderReturn", "return", 0));
        // Level 2 menus - 用户管理
        menuMapper.insert(menu(2031L, 2003L, now, "用户列表", 1, 0, "umsUser", "user", 0));
        menuMapper.insert(menu(2032L, 2003L, now, "管理员列表", 1, 1, "umsAdmin", "admin", 0));
        menuMapper.insert(menu(2033L, 2003L, now, "角色管理", 1, 2, "umsRole", "role", 0));
        // Level 2 menus - 营销管理
        menuMapper.insert(menu(2041L, 2004L, now, "优惠券管理", 1, 0, "smsCoupon", "coupon", 0));
        menuMapper.insert(menu(2042L, 2004L, now, "秒杀活动", 1, 1, "smsFlashPromotion", "flash", 0));
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
        // 超级管理员
        RoleDO superAdmin = new RoleDO();
        superAdmin.setId(3001L);
        superAdmin.setName("超级管理员");
        superAdmin.setDescription("系统所有者，拥有全部权限");
        superAdmin.setAdminCount(0);
        superAdmin.setCreateTime(now);
        superAdmin.setStatus(1);
        superAdmin.setSort(0);
        roleMapper.insert(superAdmin);
        // 运营专员
        RoleDO ops = new RoleDO();
        ops.setId(3002L);
        ops.setName("运营专员");
        ops.setDescription("日常运营、商品管理、营销管理");
        ops.setAdminCount(0);
        ops.setCreateTime(now);
        ops.setStatus(1);
        ops.setSort(1);
        roleMapper.insert(ops);
        // 客服专员
        RoleDO cs = new RoleDO();
        cs.setId(3003L);
        cs.setName("客服专员");
        cs.setDescription("售后处理、订单查看、用户查看");
        cs.setAdminCount(0);
        cs.setCreateTime(now);
        cs.setStatus(1);
        cs.setSort(2);
        roleMapper.insert(cs);
        // 仓管员
        RoleDO wh = new RoleDO();
        wh.setId(3004L);
        wh.setName("仓管员");
        wh.setDescription("发货管理");
        wh.setAdminCount(0);
        wh.setCreateTime(now);
        wh.setStatus(1);
        wh.setSort(3);
        roleMapper.insert(wh);
    }

    private void initRoleRelations() {
        // 超级管理员 (3001): all resources + all menus
        for (long resid = 101; resid <= 410; resid++) {
            if (resid == 300L || resid == 400L) continue; // skip gaps
            roleResourceRelationMapper.insert(relation(3001L, resid));
        }
        for (long menuid = 2001; menuid <= 2042; menuid++) {
            roleMenuRelationMapper.insert(menuRelation(3001L, menuid));
        }
        // 运营专员 (3002): pms:* + sms:* resources + 商品/营销 menus
        long[] opsResources = {101L, 102L, 103L, 104L, 105L, 106L, 107L, 108L, 109L, 110L, 111L, 112L,
                               401L, 402L, 403L, 404L, 405L, 406L, 407L, 408L, 409L, 410L};
        for (long resid : opsResources) {
            roleResourceRelationMapper.insert(relation(3002L, resid));
        }
        roleMenuRelationMapper.insert(menuRelation(3002L, 2001L)); // 商品管理
        roleMenuRelationMapper.insert(menuRelation(3002L, 2011L)); // 商品列表
        roleMenuRelationMapper.insert(menuRelation(3002L, 2012L)); // 商品分类
        roleMenuRelationMapper.insert(menuRelation(3002L, 2013L)); // 品牌管理
        roleMenuRelationMapper.insert(menuRelation(3002L, 2004L)); // 营销管理
        roleMenuRelationMapper.insert(menuRelation(3002L, 2041L)); // 优惠券
        roleMenuRelationMapper.insert(menuRelation(3002L, 2042L)); // 秒杀
        // 客服专员 (3003): oms:read, oms:return:*, ums:user:read resources + 订单/用户菜单
        long[] csResources = {201L, 204L, 205L, 206L, 305L};
        for (long resid : csResources) {
            roleResourceRelationMapper.insert(relation(3003L, resid));
        }
        roleMenuRelationMapper.insert(menuRelation(3003L, 2002L)); // 订单管理
        roleMenuRelationMapper.insert(menuRelation(3003L, 2021L)); // 订单列表
        roleMenuRelationMapper.insert(menuRelation(3003L, 2022L)); // 退货申请
        roleMenuRelationMapper.insert(menuRelation(3003L, 2003L)); // 用户管理
        roleMenuRelationMapper.insert(menuRelation(3003L, 2031L)); // 用户列表
        // 仓管员 (3004): oms:order:read/update + 订单菜单
        long[] whResources = {201L, 202L};
        for (long resid : whResources) {
            roleResourceRelationMapper.insert(relation(3004L, resid));
        }
        roleMenuRelationMapper.insert(menuRelation(3004L, 2002L)); // 订单管理
        roleMenuRelationMapper.insert(menuRelation(3004L, 2021L)); // 订单列表
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

    private void initDefaultAdmin() {
        AdminDO admin = adminMapper.selectById(1L);
        if (admin == null) {
            admin = new AdminDO();
            admin.setId(1L);
            admin.setUsername("admin");
            admin.setPassword("123456");
            admin.setNickName("超级管理员");
            admin.setStatus(1);
            admin.setCreateTime(LocalDateTime.now());
            adminMapper.insert(admin);
            log.info("Default admin (admin/123456) created.");
        } else {
            log.info("Default admin already exists, skipping creation.");
        }
        long roleRelationCount = adminRoleRelationMapper.selectCount(
                new LambdaQueryWrapper<AdminRoleRelationDO>()
                        .eq(AdminRoleRelationDO::getAdminId, 1L)
                        .eq(AdminRoleRelationDO::getRoleId, 3001L));
        if (roleRelationCount == 0) {
            AdminRoleRelationDO rel = new AdminRoleRelationDO();
            rel.setAdminId(1L);
            rel.setRoleId(3001L);
            adminRoleRelationMapper.insert(rel);
            log.info("Default admin role relation ensured.");
        }
    }
}
