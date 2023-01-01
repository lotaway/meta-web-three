@[TOC](Redis基础-数据类型和常见问题解决方案)

# 介绍

Redis是一个主要作用于应用程序与数据库之间，作为数据缓存层存在，通过存储在内存中高并发读写和可设置过期时间的非关系型数据库。
接下来是数据类型

# 数据类型

## string 字符串

这是最常用的数据类型，直接通过get、set即可读取和存储，通过del进行删除

```cmd
set [key] [value]
```

如：

```cmd
set dbversion 1.0.0
```

该方法无论输入值是什么类型都会作为字符串存储，读取字符串只需要使用命令：

```cmd
get [key]
```

如：

```cmd
get dbversion
```

以上方法即可读出存储的内容，若没有设置过则是null
判断是否存在字符串，返回1或0：

```cmd
exist [key]
```

删除字符串：

```cmd
del [key]
```

## list 列表

列表是含有排序的两端皆可存读的结构
从左侧存入一个值：

```cmd
lpush [key] [value]
```

从右侧存入一个值：

```cmd
rpush [key] [value]
```

从左侧拿出一个值：

```cmd
lpop [key]
```

从右侧拿出一个值：

```cmd
rpop [key]
```

根据输入索引值范围读取值，通过输入[0,-1]读取所有值：

```cmd
lrange [key] [startIndex] [endIndex]
```

获取列表长度：

```cmd
llen [key]
```

删除列表中与对应值`value`相等的，`count`绝对值代表总删除数量，正负代表从头还是从尾部开始查找：

```cmd
lrem [key] [count] [value]
```

## hash 哈希值

哈希值即是通过唯一键找到对应的值，值可能有重复，但是键绝对没有重复。
设置哈希值：

```cmd
hset [key] [hashkey] [value]
```

根据键名读取哈希值：

```cmd
hget [key] [hashkey]
```

读取所有哈希值：

```cmd
hgetall [key]
```

判断是否存在哈希值：

```cmd
hexist [key] [hashkey]
```

删除哈希值：

```cmd
hdel [key] [hashkey]
```

## set 无序集合，值不可重复

添加值：

```cmd
sadd [key] [value]
```

读取所有值：

```cmd
smembers [key]
```

删除值：

```cmd
 srem [key] [value]
```

## zset / sorted set 有序集合

添加值，`score`数值将作为排序依据，默认从小到大排序：

```cmd
zadd [key] [score] [value]
```

根据输入的索引值范围从小到大获取值，[0,-1]将读取所有值：

```cmd
zrange [key] [startIndex] [endIndex] withscores
```

根据输入的索引值范围从大到小获取值，[0,-1]将读取所有值：

```cmd
zrevrange [key] [startIndex] [endIndex] withscores
```

根据分数范围获取值：

```cmd
zrangebyscore [key] [startScore] [endScore] withscores
```

删除值：

```cmd
zrem [key] [value]
```

# 列出所有键名

```cmd
keys *
```

# Redis常见问题

## 缓存穿透

缓存穿透即如果在redis找不到缓存时，会直接访问数据库获取数据，但此时如果数据库也没有数据，而程序又因为请求不断重复这个过程，就会导致数据库访问量飙升。
解决方案：

* 对数据库空值进行缓存，这样后续空值不会访问数据库
* 启用监控黑名单，对疑似黑客攻击的端口进行限制和要求验证
* 使用布隆过滤器（后续统一讲解）
* 增加规则拦截，例如不满足uuid规则即视为无效访问

## 缓存击穿

类似缓存穿透，即高并发对空值数据访问，（原因可能是该数据为热门数据，例如活动抢购），导致就算第一次请求后设置空值或者因为缓存过期，没法阻止这种高频访问。

## 缓存雪崩

大量数据同时过期导致数据库访问量骤增。

* 调整过期时间，通过随机数对不同数据设置不同过期时间，提前缓存热门和根据数据实时热门情况调整过期时间，这样数据不会大批量同一时间过期
* 多级缓存，Nginx+redis+其他缓存
* 使用分布锁或队列，查不到数据时就加上锁，禁止其他人继续访问，强制等待

## redis服务宕机

* 削峰
* 熔断
* 集群

# 布隆过滤器

利用多个hash函数对数据进行计算，得出哈希值存放到二进制数组里，每次查询都先经过布隆过滤器判断数据是否不存在，不存在则直接返回。虽然不同数据也可能有重复的哈希值，这样数据量一多就导致判断不正确，但至少能过滤掉大量绝对不存在的数据访问请求。