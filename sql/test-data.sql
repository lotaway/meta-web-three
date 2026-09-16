-- ====================================================================================================
-- MetaWebThree - Test Data Seed Script
-- ====================================================================================================
-- Purpose:  Populate the mall + user services with realistic test data so that UI
--           showcase, admin panel browsing, and automated end-to-end (order flow)
--           tests can run immediately after `init_db.sh` creates the schema.
--
-- Database:  PostgreSQL  (all schemas share a single database)
--
-- Usage:
--   1. Run `init_db.sh` (or `docker compose up`) first to create all tables via schema.sql files.
--   2. In a psql shell connected to the same database, run:
--         \i server/test-data.sql
--   -- or --
--         psql "$DB_URL" -f server/test-data.sql
--
-- Notes:
--   * IDs are hard-coded in the 1 000 000 000 000 000 range (start with 1 quadrillion)
--     so they never clash with Java `IdWorker` (snowflake) runtime-generated IDs.
--   * Tenant ID 1 is used throughout; change `TENANT_ID` below if needed.
--   * Passwords use the {noop} prefix (plain text) for demo purposes only.
-- ====================================================================================================

\set TENANT_ID 1

-- ------------------------------------------------------------------
-- 1. USER SERVICE  (mall-domain/user-service)
-- ------------------------------------------------------------------

-- 1.1  Member levels
INSERT INTO tb_member_level (id, name, growth_point, default_status, free_freight_point,
                             comment_growth_point, priviledge_free_freight, priviledge_sign_in,
                             priviledge_comment, priviledge_promotion, priviledge_member_price,
                             priviledge_birthday, note)
VALUES
    (100000, '普通会员',  0,     1, 0.00, 5, 0, 0, 0, 0, 0, 0, '默认等级'),
    (100001, '银牌会员',  500,    0, 5.00, 10, 1, 1, 1, 0, 0, 0, '消费满 500 成长值'),
    (100002, '金牌会员',  2000,   0, 10.00, 20, 1, 1, 1, 1, 1, 1, '消费满 2000 成长值'),
    (100003, '钻石会员',  5000,   0, 20.00, 30, 1, 1, 1, 1, 1, 1, '消费满 5000 成长值');

-- 1.2  Integration consume settings
INSERT INTO tb_integration_consume_setting (id, deduction_per_amount, max_percent_per_order, use_unit, coupon_status)
VALUES (100000, 1, 20, 100, 1);

-- 1.3  Users (customers) — password is BCrypt("{noop}123456" for demo clarity)
INSERT INTO tb_user (id, username, password, nickname, avatar, email, phone, status, gender, birthday,
                     city, job, personalized_signature, integration, growth, member_level_id, source_type)
VALUES
    (1000000000000000, 'testuser1', '{noop}123456', '测试用户1', 'https://i.pravatar.cc/150?u=1', 'test1@example.com', '13800138000', 1, 1, '1990-01-15',
     '北京', '软件工程师', '热爱科技，追求极致', 500,  500,  100001, 0),
    (1000000000000001, 'testuser2', '{noop}123456', '测试用户2', 'https://i.pravatar.cc/150?u=2', 'test2@example.com', '13800138001', 1, 2, '1992-08-22',
     '上海', '产品经理',   '代码改变世界',         2000, 2000, 100002, 0),
    (1000000000000002, 'testuser3', '{noop}123456', '测试用户3', 'https://i.pravatar.cc/150?u=3', 'test3@example.com', '13800138002', 1, 0, '1985-12-03',
     '广州', '设计师',     '创意无边界',           5000, 5000, 100003, 0);

-- 1.4  Member receive addresses (one default address per user)
INSERT INTO tb_member_receive_address (id, member_id, name, phone_number, default_status, post_code,
                                      province, city, region, detail_address)
VALUES
    (1000000000000000, 1000000000000000, '测试用户1', '13800138000', 1, '100000', '北京市', '北京市', '东城区',  '东城根南路 2 号'),
    (1000000000000001, 1000000000000001, '测试用户2', '13800138001', 1, '200000', '上海市', '上海市', '徐汇区',  '交大路 100 号'),
    (1000000000000002, 1000000000000002, '测试用户3', '13800138002', 1, '518000', '广东省', '广州市', '天河区',  '天河路 1 号');

-- 1.5  Admin accounts (for backstage-admin login)
INSERT INTO tb_admin (id, username, password, icon, email, nick_name, note, status)
VALUES
    (1000000000000000, 'admin', '{noop}admin123', 'https://i.pravatar.cc/80?u=admin', 'admin@metawebthree.com', '超级管理员',  '系统初始化管理员', 1),
    (1000000000000001, 'ops',   '{noop}ops12345',  'https://i.pravatar.cc/80?u=ops',   'ops@metawebthree.com',   '运营管理员',   '业务运营管理员',   1);

-- 1.6  Roles
INSERT INTO tb_role (id, name, description, admin_count, status, sort)
VALUES
    (1000000000000000, 'ROLE_SUPER', '超级管理员', 1, 1, 0),
    (1000000000000001, 'ROLE_ADMIN', '管理员',     1, 1, 1),
    (1000000000000002, 'ROLE_OPER',  '运营人员',   1, 1, 2);

-- 1.7  Admin-role relations
INSERT INTO tb_admin_role_relation (id, admin_id, role_id)
VALUES
    (1000000000000000, 1000000000000000, 1000000000000000),  -- admin → ROLE_SUPER
    (1000000000000001, 1000000000000001, 1000000000000002);  -- ops   → ROLE_OPER

-- 1.8  Login logs
INSERT INTO tb_member_login_log (id, member_id, ip, city, login_type, province)
VALUES
    (1000000000000000, 1000000000000000, '192.168.1.100', '北京', 1, '北京'),
    (1000000000000001, 1000000000000001, '192.168.1.101', '上海', 1, '上海'),
    (1000000000000002, 1000000000000002, '192.168.1.102', '广州', 1, '广东');

-- ------------------------------------------------------------------
-- 2. PRODUCT SERVICE  (mall-domain/product-service)
-- ------------------------------------------------------------------

-- 2.1  Brands
INSERT INTO tb_brand (id, name, first_letter, sort, factory_status, show_status, product_count,
                      product_comment_count, logo, big_pic, brand_story)
VALUES
    (1000000000000000, 'Apple', 'A', 0, 1, 1, 3, 58, 'https://img.example.com/brand/apple.png',
     'https://img.example.com/brand/apple_big.png', '品质生活，从 Apple 开始'),
    (1000000000000001, '华为',  'H', 1, 1, 1, 3, 42, 'https://img.example.com/brand/huawei.png',
     'https://img.example.com/brand/huawei_big.png', '创新引领未来'),
    (1000000000000002, '小米',  'X', 2, 0, 1, 2, 18, 'https://img.example.com/brand/xiaomi.png',
     'https://img.example.com/brand/xiaomi_big.png', '感智生活，唯我为极'),
    (1000000000000003, 'Nike',  'N', 3, 0, 1, 1, 30, 'https://img.example.com/brand/nike.png',
     'https://img.example.com/brand/nike_big.png', 'Just Do It'),
    (1000000000000004, '乐烬',  'L', 4, 0, 1, 1,  6, 'https://img.example.com/brand/lejin.png',
     'https://img.example.com/brand/lejin_big.png', '发烟的乐趣');

-- 2.2  Product categories — hierarchical (parent → children)
INSERT INTO tb_product_category (id, parent_id, name, level, product_count, product_unit, nav_status,
                                 show_status, sort, icon, keywords, description)
VALUES
    -- Level 0 (root)
    (1000000000000000, 0, '电子产品',  0, 8, '件', 1, 1, 0, 'https://img.example.com/cate/electronics.png',
     '手机,电脑,wearable', '电子产品分类'),
    (1000000000000001, 0, '运动休闲',  0, 4, '件', 1, 1, 1, 'https://img.example.com/cate/sports.png',
     '球鞋,运动装', '运动休闲分类'),
    (1000000000000002, 0, '生活用品',  0, 1, '件', 1, 1, 2, 'https://img.example.com/cate/lifestyle.png',
     '生活日用,厨房', '生活用品分类'),

    -- Level 1 (children of Electronics)
    (1000000000000003, 1000000000000000, '智能手机', 1, 4, '台', 0, 1, 0, 'https://img.example.com/cate/phone.png',
     '手机,智能机', '智能手机'),
    (1000000000000004, 1000000000000000, '笔记本电脑', 1, 2, '台', 0, 1, 1, 'https://img.example.com/cate/laptop.png',
     '笔记本,电脑', '笔记本电脑'),
    (1000000000000005, 1000000000000000, '耳机音响', 1, 2, '个', 0, 1, 2, 'https://img.example.com/cate/audio.png',
     '耳机,音响', '耳机音响'),

    -- Level 1 (children of Sports)
    (1000000000000006, 1000000000000001, '跑步鞋', 1, 2, '双', 0, 1, 0, 'https://img.example.com/cate/running.png',
     '跑鞋,运动鞋', '跑步鞋'),
    (1000000000000007, 1000000000000001, '运动衣服', 1, 2, '件', 0, 1, 1, 'https://img.example.com/cate/sport-cloth.png',
     '运动服,背心', '运动衣服'),

    -- Level 1 (children of Lifestyle)
    (1000000000000008, 1000000000000002, '生活电器', 1, 1, '台', 0, 1, 0, 'https://img.example.com/cate/appliance.png',
     '生活电器', '生活电器');

-- 2.3  Product attribute categories
INSERT INTO tb_product_attribute_category (id, name, attribute_count, param_count)
VALUES
    (1000000000000000, '手机', 5, 5),
    (1000000000000001, '笔记本', 4, 4),
    (1000000000000002, '跑鞋', 4, 3),
    (1000000000000003, '运动衣', 3, 3);

-- 2.4  Product category-attribute relations
INSERT INTO tb_product_category_attribute_relation (id, product_category_id, product_attribute_id)
VALUES
    (1000000000000000, 1000000000000003, 1000000000000000),
    (1000000000000001, 1000000000000004, 1000000000000001),
    (1000000000000002, 1000000000000006, 1000000000000002),
    (1000000000000003, 1000000000000007, 1000000000000003);

-- 2.5  Products
INSERT INTO tb_product (id, brand_id, product_category_id, feight_template_id, product_attribute_category_id,
                        name, pic, product_sn, delete_status, publish_status, new_status, recommand_status,
                        verify_status, sort, sale, price, promotion_price, gift_growth, gift_point, use_point_limit,
                        sub_title, description, original_price, stock, low_stock, unit, weight, service_ids,
                        keywords, note, detail_title, detail_desc, brand_name, product_category_name, tenant_id)
VALUES
    (1000000000000000, 1000000000000000, 1000000000000003, NULL, 1000000000000000,
     'iPhone 16 Pro Max', 'https://img.example.com/product/iphone16promax.jpg',
     'SKU-IPHONE-16PROMAX-001', 0, 1, 1, 1, 1, 0, 888, 8999.00, 8999.00, 100, 100, 0,
     '6.9 超大屏 | 钛金色', 'Apple iPhone 16 Pro Max 256GB 钛金色', 9999.00, 200, 20,
     '台', 0.22, '1,2', 'iPhone 16 Pro Max', '热销爆款', 'iPhone 16 Pro Max 全面介绍',
     'Apple iPhone 16 Pro Max 拥有 A18 Pro 晶片', 'Apple', '智能手机', :TENANT_ID),

    (1000000000000001, 1000000000000001, 1000000000000003, NULL, 1000000000000000,
     '华为 Mate 70 Pro+', 'https://img.example.com/product/huawei-mate70proplus.jpg',
     'SKU-HUAWEI-MATE70PROPLUS-001', 0, 1, 1, 1, 1, 0, 655, 6999.00, 6499.00, 80, 80, 0,
     '40MP 梅塞德斯-曼苏罗影像 | 星河纪念版', '华为 Mate 70 Pro+ 桦石紫 512GB',
     7998.00, 160, 15, '台', 0.20, '1,2', '华为 Mate 70', '热销爆款', '华为 Mate 70 Pro+ 全面介绍',
     '华为 Mate 70 Pro+ 拥有哈勃影像系统', '华为', '智能手机', :TENANT_ID),

    (1000000000000002, 1000000000000000, 1000000000000004, NULL, 1000000000000001,
     'MacBook Pro 14', 'https://img.example.com/product/macbook-pro-14.jpg',
     'SKU-MBP-14-2024-001', 0, 1, 1, 1, 1, 0, 520, 15999.00, 15999.00, 200, 200, 0,
     'M4 Pro | 18GB 内存 | 512GB SSD', 'Apple MacBook Pro 14 M4 Pro 18GB 512GB',
     17999.00, 80, 10, '台', 1.4, '1,2', 'MacBook Pro', '热销爆款', 'MacBook Pro 14 M4 Pro 全面介绍',
     'Apple MacBook Pro 14寸 M4 Pro 处理器', 'Apple', '笔记本电脑', :TENANT_ID),

    (1000000000000003, 1000000000000001, 1000000000000004, NULL, 1000000000000001,
     'HUAWEI MateBook X Pro 14', 'https://img.example.com/product/matebook-xpro-14.jpg',
     'SKU-MATEBOOK-XPRO-14-001', 0, 1, 0, 1, 1, 0, 320, 8999.00, 7999.00, 100, 100, 0,
     '14寸 2.8K 屏 | 全面屏设计', '华为 MateBook X Pro 14 i7-1360P 16GB 512GB',
     10999.00, 60, 5, '台', 1.18, '1,2', 'MateBook X Pro', '热销爆款', 'HUAWEI MateBook X Pro 14 全面介绍',
     '华为 MateBook X Pro 14寸笔记本', '华为', '笔记本电脑', :TENANT_ID),

    (1000000000000004, 1000000000000003, 1000000000000006, NULL, 1000000000000002,
     'Nike Air Zoom Pegasus 41', 'https://img.example.com/product/nike-pegasus41.jpg',
     'SKU-NIKE-PEGASUS41-001', 0, 1, 1, 1, 1, 0, 999, 799.00, 799.00, 30, 30, 0,
     'ZoomX 气垫鞋底 | 轻便透气', 'Nike Air Zoom Pegasus 41 男款黑/红', 899.00, 240, 25,
     '双', 0.26, '1,2', 'Nike 跑鞋', '热销爆款', 'Nike Air Zoom Pegasus 41 全面介绍',
     'Nike Air Zoom Pegasus 41 采用 ZoomX 气垫', 'Nike', '跑步鞋', :TENANT_ID),

    (1000000000000005, 1000000000000003, 1000000000000006, NULL, 1000000000000002,
     'Nike Dri-FIT 跑步服套装', 'https://img.example.com/product/nike-dri-fit-set.jpg',
     'SKU-NIKE-DRIFIT-SET-001', 0, 1, 0, 0, 1, 0, 420, 399.00, 299.00, 20, 20, 0,
     '干湿透气 | 四季款', 'Nike Dri-FIT 黑/蓝 M', 499.00, 160, 15, '套', 0.32, '1,2',
     'Nike 运动衣', '热销爆款', 'Nike Dri-FIT 跑步服套装 全面介绍',
     'Nike Dri-FIT 干湿透气运动服', 'Nike', '跑步鞋', :TENANT_ID),

    (1000000000000006, 1000000000000004, 1000000000000008, NULL, 1000000000000003,
     '华为智能烟盒 EL X2', 'https://img.example.com/product/huawei-lighter-x2.jpg',
     'SKU-HUAWEI-LIGHTER-X2-001', 0, 1, 0, 0, 1, 0, 156, 199.00, 199.00, 10, 10, 0,
     '智能温控 | 可定时点燃', '华为智能烟盒 EL X2 星空黑', 268.00, 100, 10,
     '个', 0.08, '1,2', '智能烟盒', '热销爆款', '华为智能烟盒 EL X2 全面介绍',
     '华为智能烟盒 EL X2 支持 App 控制', '华为', '生活电器', :TENANT_ID),

    (1000000000000007, 1000000000000002, 1000000000000005, NULL, 1000000000000000,
     '小米 14', 'https://img.example.com/product/xiaomi-14.jpg',
     'SKU-XIAOMI-14-001', 0, 1, 0, 0, 1, 0, 777, 3999.00, 3699.00, 60, 60, 0,
     '骁龙 8 Gen 3 | 哈苏影像', '小米 14 银河白 512GB', 4299.00, 180, 15,
     '台', 0.17, '1,2', '小米 14', '热销爆款', '小米 14 全面介绍',
     '小米 14 搭载骁龙 8 Gen 3 处理器', '小米', '智能手机', :TENANT_ID),

    (1000000000000008, 1000000000000001, 1000000000000007, NULL, 1000000000000002,
     'Nike Air Force 1 Low', 'https://img.example.com/product/nike-af1-low.jpg',
     'SKU-NIKE-AF1-LOW-001', 0, 1, 0, 0, 1, 0, 666, 699.00, 699.00, 30, 30, 0,
     '经典复刻 | 全白配色', 'Nike Air Force 1 Low 全白', 799.00, 220, 22,
     '双', 0.25, '1,2', 'Nike AF1', '热销爆款', 'Nike Air Force 1 Low 全面介绍',
     'Nike Air Force 1 是百搭球鞋', 'Nike', '跑步鞋', :TENANT_ID);

-- 2.6  SKU stocks (2 SKUs for each product: standard + premium variant)
INSERT INTO tb_sku_stock (id, product_id, sku_code, price, stock, low_stock, pic, sale,
                          promotion_price, lock_stock, sp_data, tenant_id)
VALUES
    -- iPhone 16 Pro Max
    (1000000000000000, 1000000000000000, 'SKU-IPH16PM-BLK-256G', 8999.00, 50, 10,
     'https://img.example.com/product/iphone16promax-black.jpg', 300, 8999.00, 0,
     '{"color":"钛晴色","storage":"256GB"}', :TENANT_ID),
    (1000000000000001, 1000000000000000, 'SKU-IPH16PM-SIL-256G', 8999.00, 50, 10,
     'https://img.example.com/product/iphone16promax-silver.jpg', 250, 8999.00, 0,
     '{"color":"钯金色","storage":"256GB"}', :TENANT_ID),

    -- Huawei Mate 70 Pro+
    (1000000000000002, 1000000000000001, 'SKU-HWMP-BLK-512G', 6499.00, 40, 10,
     'https://img.example.com/product/huawei-mate70proplus-black.jpg', 180, 6499.00, 0,
     '{"color":"桦石紫","storage":"512GB"}', :TENANT_ID),
    (1000000000000003, 1000000000000001, 'SKU-HWMP-SIL-512G', 6499.00, 40, 10,
     'https://img.example.com/product/huawei-mate70proplus-silver.jpg', 120, 6499.00, 0,
     '{"color":"珩玉白","storage":"512GB"}', :TENANT_ID),

    -- MacBook Pro 14
    (1000000000000004, 1000000000000002, 'SKU-MBP14-SP-512G', 15999.00, 30, 5,
     'https://img.example.com/product/macbook-pro-14-silver.jpg', 90, 15999.00, 0,
     '{"color":"银色","storage":"512GB"}', :TENANT_ID),
    (1000000000000005, 1000000000000002, 'SKU-MBP14-SP-1TB', 17999.00, 20, 5,
     'https://img.example.com/product/macbook-pro-14-silver-1tb.jpg', 40, 17999.00, 0,
     '{"color":"银色","storage":"1TB"}', :TENANT_ID),

    -- Huawei MateBook X Pro 14
    (1000000000000006, 1000000000000003, 'SKU-MBPXRD-CL-512G', 7999.00, 30, 5,
     'https://img.example.com/product/matebook-xpro-cl-512g.jpg', 75, 7999.00, 0,
     '{"color":"紫晶色","storage":"512GB"}', :TENANT_ID),

    -- Nike Pegasus 41
    (1000000000000007, 1000000000000004, 'SKU-NIKE-PG41-BKRED-41', 799.00, 60, 15,
     'https://img.example.com/product/nike-pegasus41-blackred.jpg', 220, 799.00, 0,
     '{"color":"黑/红","size":"41"}', :TENANT_ID),
    (1000000000000008, 1000000000000004, 'SKU-NIKE-PG41-BKRED-42', 799.00, 60, 15,
     'https://img.example.com/product/nike-pegasus41-blackred-42.jpg', 190, 799.00, 0,
     '{"color":"黑/红","size":"42"}', :TENANT_ID),

    -- Nike Dri-FIT
    (1000000000000009, 1000000000000005, 'SKU-NIKE-DRIFIT-L', 299.00, 40, 10,
     'https://img.example.com/product/nike-dri-fit-l.jpg', 95, 299.00, 0,
     '{"color":"黑/蓝","size":"L"}', :TENANT_ID),

    -- Huawei Lighter EL X2
    (1000000000000010, 1000000000000006, 'SKU-HUAWEI-LT-X2-BLK', 199.00, 30, 5,
     'https://img.example.com/product/huawei-lighter-x2-black.jpg', 60, 199.00, 0,
     '{"color":"星空黑"}', :TENANT_ID),

    -- Xiaomi 14
    (1000000000000011, 1000000000000007, 'SKU-XM14-SLV-512G', 3699.00, 45, 10,
     'https://img.example.com/product/xiaomi-14-silver.jpg', 150, 3699.00, 0,
     '{"color":"银河白","storage":"512GB"}', :TENANT_ID),

    -- Nike Air Force 1 Low
    (1000000000000012, 1000000000000008, 'SKU-NIKE-AF1LOW-WHT-41', 699.00, 55, 12,
     'https://img.example.com/product/nike-af1low-white.jpg', 130, 699.00, 0,
     '{"color":"全白","size":"41"}', :TENANT_ID);

-- 2.7  Update product stock numbers to match sum of SKUs
UPDATE tb_product SET
    stock  = (SELECT COALESCE(SUM(stock), 0) FROM tb_sku_stock s WHERE s.product_id = tb_product.id),
    sale   = (SELECT COALESCE(SUM(sale), 0)  FROM tb_sku_stock s WHERE s.product_id = tb_product.id)
WHERE id IN (1000000000000000, 1000000000000001, 1000000000000002,
             1000000000000003, 1000000000000004, 1000000000000005,
             1000000000000006, 1000000000000007, 1000000000000008);

-- 2.8  Home recommendation data
INSERT INTO tb_home_brand (id, brand_id, brand_name, recommend_status, sort) VALUES
    (1000000000000000, 1000000000000000, 'Apple', 1, 0),
    (1000000000000001, 1000000000000001, '华为', 1, 1),
    (1000000000000002, 1000000000000003, 'Nike', 1, 2);

INSERT INTO tb_home_recommend_product (id, product_id, product_name, recommend_status, sort) VALUES
    (1000000000000000, 1000000000000000, 'iPhone 16 Pro Max', 1, 0),
    (1000000000000001, 1000000000000001, '华为 Mate 70 Pro+', 1, 1),
    (1000000000000002, 1000000000000002, 'MacBook Pro 14', 1, 2),
    (1000000000000003, 1000000000000004, 'Nike Air Zoom Pegasus 41', 1, 3);

INSERT INTO tb_home_new_product (id, product_id, product_name, recommend_status, sort) VALUES
    (1000000000000000, 1000000000000004, 'Nike Air Force 1 Low', 1, 0),
    (1000000000000001, 1000000000000006, '华为智能烟盒 EL X2', 1, 1);

INSERT INTO tb_home_advertise (id, name, type, pic, start_time, end_time, status, click_count, order_count, url, note, sort) VALUES
    (1000000000000000, 'iPhone 16 Pro 推荐', 1, 'https://img.example.com/ad/iphone16.jpg',
     NOW() - INTERVAL '7 day', NOW() + INTERVAL '7 day', 1, 1200, 340,
     '/product/1000000000000000', '首页主推广告', 0);

-- ------------------------------------------------------------------
-- 3. CART SERVICE  (mall-domain/cart-service)
-- ------------------------------------------------------------------
INSERT INTO oms_cart_item (id, product_id, product_sku_id, member_id, quantity, price,
                            product_pic, product_name, product_sub_title, product_sku_code,
                            member_nickname, product_category_id, product_brand, product_sn,
                            product_attr, tenant_id)
VALUES
    (1000000000000000, 1000000000000000, 1000000000000000, 1000000000000000, 1, 8999.00,
     'https://img.example.com/product/iphone16promax.jpg',
     'iPhone 16 Pro Max', '6.9 超大屏 | 钛金色', 'SKU-IPH16PM-BLK-256G',
     '测试用户1', 1000000000000003, 'Apple', 'SKU-IPHONE-16PROMAX-001',
     '{"color":"钛晴色","storage":"256GB"}', :TENANT_ID);

-- ------------------------------------------------------------------
-- 4. PROMOTION SERVICE  (mall-domain/promotion-service)
-- ------------------------------------------------------------------

-- 4.1  Coupons
INSERT INTO tb_coupon (id, type, name, platform, count, amount, per_limit, min_point,
                        start_time, end_time, use_type, note, publish_count, use_count,
                        receive_count, enable_time, code, member_level, tenant_id)
VALUES
    (1000000000000000, 0, '满 100 立减 10 元', 0, 50000, 10.00, 1, 100.00,
     NOW() - INTERVAL '1 day', NOW() + INTERVAL '30 day', 0, '全站通用满减券',
     0, 0, 0, NOW() - INTERVAL '1 day', 'COUPON_ALL_001', 0, :TENANT_ID),

    (1000000000000001, 1, '钻石会员专享满 500 减 50', 0, 10000, 50.00, 1, 500.00,
     NOW() - INTERVAL '1 day', NOW() + INTERVAL '30 day', 0, '钻石会员专享',
     0, 0, 0, NOW() - INTERVAL '1 day', 'COUPON_DIAMOND_001', 3, :TENANT_ID),

    (1000000000000002, 3, '新用户注册专享 5 元券', 0, 20000, 5.00, 1, 0.00,
     NOW() - INTERVAL '1 day', NOW() + INTERVAL '30 day', 0, '注册即得',
     0, 0, 0, NOW() - INTERVAL '1 day', 'COUPON_NEW_001', 0, :TENANT_ID);

-- 4.2  Coupon history — pre-assign a coupon to testuser3 (diamond member)
INSERT INTO tb_coupon_history (id, coupon_id, member_id, coupon_code, member_nickname, get_type,
                               use_status, use_time, order_id, order_sn, tenant_id)
VALUES
    (1000000000000000, 1000000000000001, 1000000000000002, 'COUPON_DIAMOND_001_001',
     '测试用户3', 0, 0, NULL, NULL, NULL, :TENANT_ID);

-- 4.3  Flash promotion
INSERT INTO tb_flash_promotion (id, title, start_date, end_date, status, create_time, tenant_id)
VALUES
    (1000000000000000, '双 11 预售', CURRENT_DATE - INTERVAL '1 day',
     CURRENT_DATE + INTERVAL '1 day', 1, NOW(), :TENANT_ID);

INSERT INTO tb_flash_promotion_session (id, name, start_time, end_time, status, create_time, tenant_id)
VALUES
    (1000000000000000, '双 11 预售场', '00:00:00', '23:59:59', 1, NOW(), :TENANT_ID);

INSERT INTO tb_flash_promotion_product_relation (id, flash_promotion_id, flash_promotion_session_id,
                                                  product_id, flash_promotion_price,
                                                  flash_promotion_count, flash_promotion_limit, sort, tenant_id)
VALUES
    (1000000000000000, 1000000000000000, 1000000000000000,
     1000000000000000, 7999.00, 10, 1, 0, :TENANT_ID);

-- ------------------------------------------------------------------
-- 5. ORDER SERVICE  (mall-domain/order-service)
-- ------------------------------------------------------------------

-- 5.1  Company address
INSERT INTO tb_company_address (id, address_name, send_status, receive_status, name, phone,
                                province, city, region, detail_address, tenant_id)
VALUES
    (1000000000000000, '默认发货仓库', 1, 1, 'MetaWeb 仓库管理员', '010-12345678',
     '北京市', '北京市', '朝阳区', '朝阳路 100 号仓储中心', :TENANT_ID);

-- 5.2  Orders — 3 sample orders covering different statuses
--      Status: 0=待支付, 1=已支付, 2=已发货, 3=已完成, 4=已取消, 5=无效
INSERT INTO tb_order (id, member_id, coupon_id, order_sn, create_time, member_username,
                       total_amount, pay_amount, freight_amount, promotion_amount,
                       integration_amount, coupon_amount, discount_amount,
                       pay_type, source_type, status, order_type,
                       receiver_name, receiver_phone, receiver_post_code,
                       receiver_province, receiver_city, receiver_region, receiver_detail_address,
                       note, confirm_status, delete_status, member_receive_address_id,
                       use_integration, payment_time, delivery_time,
                       receive_time, comment_time, modify_time, tenant_id)
VALUES
    -- Order #1 — 已完成 (Completed)
    (1000000000000000, 1000000000000000, NULL, '20240000000001',
     NOW() - INTERVAL '5 day', 'testuser1',
     9798.00, 9798.00, 0.00, 0.00, 0.00, 0.00, 0.00,
     1, 0, 3, 0,
     '测试用户1', '13800138000', '100000',
     '北京市', '北京市', '东城区', '东城根南路 2 号',
     '请尽快发货', 1, 0, 1000000000000000, 100,
     NOW() - INTERVAL '5 day', NOW() - INTERVAL '4 day',
     NOW() - INTERVAL '3 day', NOW() - INTERVAL '2 day', NOW() - INTERVAL '2 day', :TENANT_ID),

    -- Order #2 — 待支付 (Pending payment)
    (1000000000000001, 1000000000000001, NULL, '20240000000002',
     NOW() - INTERVAL '1 hour', 'testuser2',
     15999.00, 15999.00, 0.00, 0.00, 0.00, 0.00, 0.00,
     1, 0, 0, 0,
     '测试用户2', '13800138001', '200000',
     '上海市', '上海市', '徐汇区', '交大路 100 号',
     NULL, 0, 0, 1000000000000001, 200,
     NULL, NULL,
     NULL, NULL, NOW() - INTERVAL '1 hour', :TENANT_ID),

    -- Order #3 — 已取消 (Cancelled)
    (1000000000000002, 1000000000000000, NULL, '20240000000003',
     NOW() - INTERVAL '3 day', 'testuser1',
     799.00, 799.00, 0.00, 0.00, 0.00, 0.00, 0.00,
     1, 0, 4, 0,
     '测试用户1', '13800138000', '100000',
     '北京市', '北京市', '东城区', '东城根南路 2 号',
     '已取消订单用于测试', 0, 0, 1000000000000000, 30,
     NULL, NULL,
     NULL, NULL, NOW() - INTERVAL '3 day', :TENANT_ID);

-- 5.3  Order items
INSERT INTO tb_order_item (id, order_id, order_sn, product_id, product_pic, product_name,
                            product_brand, product_sn, product_price, product_quantity,
                            product_sku_id, product_sku_code, product_category_id,
                            promotion_name, promotion_amount, coupon_amount, integration_amount,
                            real_amount, gift_integration, gift_growth, product_attr, tenant_id)
VALUES
    -- Order #1 — items
    (1000000000000000, 1000000000000000, '20240000000001', 1000000000000000,
     'https://img.example.com/product/iphone16promax.jpg', 'iPhone 16 Pro Max',
     'Apple', 'SKU-IPHONE-16PROMAX-001', 8999.00, 1,
     1000000000000000, 'SKU-IPH16PM-BLK-256G', 1000000000000003,
     NULL, NULL, NULL, NULL, 8999.00, 100, 100,
     '{"color":"钛晴色","storage":"256GB"}', :TENANT_ID),

    (1000000000000001, 1000000000000000, '20240000000001', 1000000000000004,
     'https://img.example.com/product/nike-pegasus41.jpg', 'Nike Air Zoom Pegasus 41',
     'Nike', 'SKU-NIKE-PEGASUS41-001', 799.00, 1,
     1000000000000007, 'SKU-NIKE-PG41-BKRED-41', 1000000000000006,
     NULL, NULL, NULL, NULL, 799.00, 30, 30,
     '{"color":"黑/红","size":"41"}', :TENANT_ID),

    -- Order #2 — items
    (1000000000000002, 1000000000000001, '20240000000002', 1000000000000002,
     'https://img.example.com/product/macbook-pro-14.jpg', 'MacBook Pro 14',
     'Apple', 'SKU-MBP-14-2024-001', 15999.00, 1,
     1000000000000004, 'SKU-MBP14-SP-512G', 1000000000000004,
     NULL, NULL, NULL, NULL, 15999.00, 200, 200,
     '{"color":"银色","storage":"512GB"}', :TENANT_ID),

    -- Order #3 — items
    (1000000000000003, 1000000000000002, '20240000000003', 1000000000000004,
     'https://img.example.com/product/nike-pegasus41.jpg', 'Nike Air Zoom Pegasus 41',
     'Nike', 'SKU-NIKE-PEGASUS41-001', 799.00, 1,
     1000000000000007, 'SKU-NIKE-PG41-BKRED-41', 1000000000000006,
     NULL, NULL, NULL, NULL, 799.00, 30, 30,
     '{"color":"黑/红","size":"41"}', :TENANT_ID);

-- 5.4  Order setting
INSERT INTO tb_order_setting (id, flash_order_overtime, normal_order_overtime, confirm_overtime,
                               finish_overtime, comment_overtime)
VALUES (1000000000000000, 1, 30, 7, 7, 30);

-- 5.5  Order return reasons
INSERT INTO tb_order_return_reason (id, name, sort, status, create_time)
VALUES
    (1000000000000000, '不符合要求', 0, 1, NOW()),
    (1000000000000001, '质量问题', 1, 1, NOW()),
    (1000000000000002, '尺寸/颜色不符', 2, 1, NOW()),
    (1000000000000003, '物流问题', 3, 1, NOW()),
    (1000000000000004, '已经买错了', 4, 1, NOW());

-- ------------------------------------------------------------------
-- 6. REVIEW SERVICE  (mall-domain/review-service)
-- ------------------------------------------------------------------
INSERT INTO review (id, order_id, order_item_id, product_id, sku_id, user_id, store_id,
                     rating, content, images, status, like_count, reply_count,
                     reply_content, create_time, update_time, tenant_id)
VALUES
    (1000000000000000, 1000000000000000, 1000000000000000, 1000000000000000,
     1000000000000000, 1000000000000000, NULL,
     5, '手机非常好，用起来特别流畅，拍照效果也很不错，强烈推荐！',
     '["https://img.example.com/review/phone1.jpg", "https://img.example.com/review/phone2.jpg"]',
     1, 10, 2, '感谢您的评价，祝您购物愉快！',
     NOW() - INTERVAL '2 day', NOW() - INTERVAL '2 day', :TENANT_ID),

    (1000000000000001, 1000000000000000, 1000000000000001, 1000000000000004,
     1000000000000007, 1000000000000000, NULL,
     4, '鞋子穿上比较舒适，但是物流有点慢',
     '["https://img.example.com/review/shoes1.jpg"]',
     1, 5, 0, NULL,
     NOW() - INTERVAL '2 day', NOW() - INTERVAL '2 day', :TENANT_ID);

-- ------------------------------------------------------------------
-- 7. AFTER-SALE SERVICE  (mall-domain/after-sale-service)
-- ------------------------------------------------------------------
-- Leave empty (no after-sales yet), ready for the user to create one during testing.

-- ------------------------------------------------------------------
-- 8. PAYMENT SERVICE  (mall-domain/payment-service)
-- ------------------------------------------------------------------
INSERT INTO Credit_Profile (user_id, base_credit_limit, current_credit_limit, credit_used,
                             risk_level, last_score, last_limit_adjustment,
                             overdue_count_3m, transaction_success_rate,
                             credit_utilization_rate, last_score_change,
                             max_adjustment_percentage, adjustment_history,
                             last_update, tenant_id)
VALUES
    (1000000000000000, 10000, 10000, 0,    'C', 65, NOW() - INTERVAL '1 day',
     0, 100.00, 0.00, 0, 0.15, '[]', NOW() - INTERVAL '1 day', :TENANT_ID),
    (1000000000000001, 10000, 10000, 0,    'C', 82, NOW() - INTERVAL '1 day',
     0, 100.00, 0.00, 0, 0.15, '[]', NOW() - INTERVAL '1 day', :TENANT_ID),
    (1000000000000002, 20000, 20000, 500,  'B', 72, NOW() - INTERVAL '2 day',
     0, 98.50, 2.50, 0, 0.15, '[]', NOW() - INTERVAL '2 day', :TENANT_ID);

-- ====================================================================================================
-- End of test-data.sql
-- ====================================================================================================
