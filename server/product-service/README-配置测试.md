# Spring Boot 配置测试方法

本文档介绍在Spring Boot应用中测试自动注入配置的多种生命周期方法，以及如何解决公共配置读取问题。

## 问题背景

在Spring Boot应用中，我们经常需要验证配置文件中的属性是否被正确读取和注入。由于静态方法无法访问非静态字段，我们需要使用Spring Boot提供的生命周期方法。

### 公共配置读取问题

在多模块项目中，每个服务模块都有自己的配置文件，但公共配置（如RocketMQ、数据库连接等）需要在多个服务间共享。常见的解决方案包括：

1. **环境变量方式**（推荐）- 通过环境变量设置公共配置
2. **配置中心** - 使用Spring Cloud Config等配置中心
3. **配置文件继承** - 通过Spring Boot的配置继承机制

## 推荐方法

### 1. ApplicationReadyEvent（最推荐）

**执行时机**: 应用完全启动后，所有Bean都已初始化完成
**优点**: 确保所有配置都已加载完成，最安全可靠
**使用场景**: 生产环境中的配置验证

```java
@Component
public static class ConfigurationTest implements ApplicationListener<ApplicationReadyEvent> {
    
    @Value("${rocketmq.client.namesrv}")
    private String namesrv;
    
    @Override
    public void onApplicationEvent(ApplicationReadyEvent event) {
        System.out.println("namesrv: " + namesrv);
    }
}
```

### 2. @PostConstruct

**执行时机**: Bean实例化后，依赖注入完成后立即执行
**优点**: 执行时机早，可以及早发现问题
**使用场景**: 开发阶段的快速验证

```java
@Component
public class ConfigurationTest {
    
    @Value("${rocketmq.client.namesrv}")
    private String namesrv;
    
    @PostConstruct
    public void test() {
        System.out.println("namesrv: " + namesrv);
    }
}
```

### 3. CommandLineRunner

**执行时机**: 应用启动完成后，在main方法返回之前执行
**优点**: 可以访问命令行参数，适合需要处理启动参数的场景
**使用场景**: 需要根据启动参数进行配置验证

```java
@Component
public static class ConfigurationTest implements CommandLineRunner {
    
    @Value("${rocketmq.client.namesrv}")
    private String namesrv;
    
    @Override
    public void run(String... args) throws Exception {
        System.out.println("namesrv: " + namesrv);
    }
}
```

## 执行顺序

1. `@PostConstruct` - 最早执行
2. `ApplicationReadyEvent` - 应用完全启动后
3. `CommandLineRunner` - 最后执行

## 配置属性默认值

使用`@Value`注解时，可以设置默认值：

```java
@Value("${rocketmq.client.namesrv:默认值}")
private String namesrv;
```

## 最佳实践

1. **生产环境**: 使用 `ApplicationReadyEvent`
2. **开发环境**: 使用 `@PostConstruct` 进行快速验证
3. **需要命令行参数**: 使用 `CommandLineRunner`
4. **配置验证**: 在测试方法中添加断言或日志记录
5. **错误处理**: 在配置读取失败时提供有意义的错误信息

## 示例输出

启动应用后，控制台会显示类似以下输出：

```
=== @PostConstruct 配置测试 ===
namesrv: localhost:9876
server.port: 8080
=== @PostConstruct 配置测试完成 ===

=== ApplicationReadyEvent 配置测试 ===
namesrv: localhost:9876
=== ApplicationReadyEvent 配置测试完成 ===

=== CommandLineRunner 配置测试 ===
namesrv: localhost:9876
=== CommandLineRunner 配置测试完成 ===
```

## 公共配置管理

### 环境变量方式（推荐）

在 `application.yml` 中使用环境变量：

```yaml
rocketmq:
  client:
    namesrv: ${ROCKETMQ_NAMESRV:192.168.43.137:9876}
```

启动时设置环境变量：

```bash
export ROCKETMQ_NAMESRV=192.168.43.137:9876
mvn spring-boot:run
```

### 使用启动脚本

项目提供了 `start-with-env.sh` 脚本：

```bash
./start-with-env.sh
```

### 配置验证

使用 `CommonConfig` 类来验证配置是否正确读取：

```java
@Component
public static class ConfigValidationListener implements ApplicationListener<ApplicationReadyEvent> {
    @Value("${rocketmq.client.namesrv:未配置}")
    private String rocketmqNamesrv;
    
    @Override
    public void onApplicationEvent(ApplicationReadyEvent event) {
        if ("未配置".equals(rocketmqNamesrv)) {
            System.err.println("警告: RocketMQ namesrv 未正确配置!");
        } else {
            System.out.println("✓ RocketMQ 配置正确");
        }
    }
}
```

## 注意事项

1. 避免在静态方法中访问非静态字段
2. 使用`@Value`注解时提供默认值，避免配置缺失导致的启动失败
3. 在生产环境中，建议使用日志框架而不是`System.out.println`
4. 配置测试代码应该在验证完成后移除或禁用
5. 公共配置建议使用环境变量方式管理，便于不同环境部署 