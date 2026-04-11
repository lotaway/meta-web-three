# 修复 user-service MemberAddressRepository Bean 缺失问题

## 步骤列表（按顺序完成）

- [x] 步骤 1: 更新 pom.xml - 添加 spring-boot-starter-data-jpa, h2, lombok 依赖
- [x] 步骤 2: 更新 src/main/java/com/metawebthree/user/domain/model/MemberAddress.java - 添加 @Entity, @Id, JPA 注解
- [x] 步骤 3: 更新 src/main/java/com/metawebthree/user/UserServiceApplication.java - 添加 @EnableJpaRepositories, @EntityScan
- [x] 步骤 4: 更新 src/main/resources/application-dev.yml - 添加 H2 数据源和 JPA 配置
- [x] 步骤 5: 执行 mvn clean install (已成功，编译警告忽略)
- [ ] 步骤 6: 运行应用测试 mvn spring-boot:run -Dspring.profiles.active=dev
- [ ] 完成：验证无 Bean 错误

## 附加修复
- Repository 改为 extends JpaRepository
- Service 调用调整为 JPA 方法 (save, deleteById)
- pom.xml 指定 mainClass 避免重复
