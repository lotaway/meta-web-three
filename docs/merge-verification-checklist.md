# 微服务合并验证清单

> 本清单基于实际可用的脚本（`run-server.sh`、`scripts/check-server-memory.sh`）制定。仓库中原计划的 `verify-all.sh` 不存在，因此以现有工具为准。

## 1. 代码合并（per-step）
- [ ] 保留原包结构（不重命名包路径）
- [ ] `@SpringBootApplication(scanBasePackages = ...)` 使用宿主 + 被合并包，避免全包扫描
- [ ] `@MapperScan(basePackages = {...}, annotationClass = Mapper.class)` 扫描所有业务包
- [ ] `db/schema.sql` 追加被合并服务的表结构
- [ ] 资源/配置按目标服务为准（不强行合并 application.yml）
- [ ] 从 `server/pom.xml`、`scripts/server-services-registry.sh` 移除旧模块
- [ ] 删除旧模块目录（git 可回滚）

## 2. 构建验证
```bash
# 目标服务构建（示例：factory-service）
mvn -pl factory-domain/factory-service -am clean install -Dmaven.test.skip=true

# 全量构建（快速回归）
mvn clean install -Dmaven.test.skip=true -T 4
```

## 3. 启动验证（单服务）
```bash
# 使用 run-server.sh 的注册表或手动启动
# 查看启动日志
tail -f logs/run-server/<service-name>.log
# 判定：出现 "Started ...Application in ... seconds"
```

## 4. 冒烟验证
- Swagger/OpenAPI：`GET http://localhost:<port>/v3/api-docs`（预期 200）
- 业务端点：按各服务 Controller 主要路径访问
- GraphQL（如有子图）：通过网关 `/graphql` 或各自端点验证
- WebSocket（如有）：连接测试

> 环境性失败（dev 库缺表 `relation "..." does not exist`、缺必填头、AWS 凭据缺失、400 缺参）不计为接线问题。

## 5. 全量回归（推荐）
```bash
# 全量构建+启动14个服务+监控存活（脚本内含内存检测与优雅停止）
bash /d/temp/opencode/regression.sh  # 备选：封装了 run-server.sh

# 或直接
./run-server.sh
# 另开终端：内存复测
./scripts/check-server-memory.sh
```

判定标准：
- [ ] `BUILD SUCCESS`
- [ ] 所有注册服务均 `STARTED`（日志中出现 Started 行）
- [ ] 监控显示 `14/14 服务存活`
- [ ] 无 `APPLICATION FAILED TO START`、`Failed to bind` 等致命错误

## 6. 网关/前端同步核对（收尾必查）
- [ ] `server/gateway/src/main/resources/application.yml` 路径按合后服务名
- [ ] `GatewayAuthConfig.routeRoles` 使用合后服务 id + 实际 Controller 路径
- [ ] `CircuitBreakerFilter.determineServiceName` 能正确识别合后服务名
- [ ] `apps/backstage-admin/src/utils/http.ts` `SERVICE_PREFIX_MAP` 覆盖新增映射（`/api/forecasting`、`/api/v1/route-optimizer`→ai-service、`/project-service`→erp-management-service）
- [ ] `apps/client/*` 硬编码服务名已替换为合后服务名
- [ ] `config/domain/*.yaml` 服务键为合后服务名
- [ ] `docs/microservice-consolidation.md`、`server/README.md` 已同步

## 7. 提交规范
- 每步通过后用英文 commit message 提交
- 提交前确认工作树干净：`git status --short` 应为空（或仅预期改动）

> 实际使用脚本清单：`run-server.sh`（全量启动）、`scripts/check-server-memory.sh`（内存报告）、`scripts/server-services-registry.sh`（服务注册表）。未提供 `verify-all.sh`，本清单已基于现有脚本调整。
