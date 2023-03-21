# SpringBoot配置

SpringBoot基于约定，有很多默认配置值，需要修改则要使用约定好的名称并写上想要设定的值。
application.properties：

```properties
server.port=8080
```

application.yml/application.yaml：

```yaml
server:
  port: 8080
```

三个配置文件都存在时，以properties中的设定为最优先级