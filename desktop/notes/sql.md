@(TOC)[SQL学习-基础课程]

# 概念

SQL是Structured Query Language 的缩写,意为“结构化查询语言”。一般代指现在计算机中的各种SQL数据库，如MySQL、MicrosoftServerSQL。

* 以下语句以MySQL为主，其他数据库在字段类型、语法上会有轻微区别，但大同小异。

# 创建

通过`create table [tableName] (Array<field>)`创建一个用户表：

```sql
--  用户表
create table `User`
(
    id         integer primary key unique auto_increment comment '用户标识',
    username   varchar(20) not null unique comment '用户名',
    nickname   varchar(20)          default '' comment '昵称',
    sex        char(2)     not null default 0 comment '性别',
    age        char(2)     not null default 0 comment '年龄',
    authorId   integer comment '作者标识',
    createTime Datetime(3) default Datetime(3),
    updateTime Datetime(3) default Datetime(3)
);
-- 作者表
create table `Author`
(
    id       integer primary key unique auto_increment comment '作者标识',
    nickname varchar(20) unique comment '昵称',
    userId   integer not null,
    constraint `Author_userId_fkey` foreign key (userId) references `User` (`id`),
    state    char(1) default 0 comment '账号状态',
    category tinyint(50) comment '写作类型',
    level    tinyint(10) comment '级别'
);
-- 书籍表
create table `Book`
(
    id        integer primary key unique auto_increment comment '书标识',
    authorId  integer     not null,
    constraint `Book_authorId_fkey` foreign key (authorId) references `Author` (`id`),
    bookName  varchar(30) not null unique comment '书名',
    state     char(1) default 0 comment '书籍状态',
    readCount integer default 0 '阅读量'
)
-- 书籍章节内容
create table `BookChapter`
(
    id        varchar(50) unique default uuid() comment '书籍章节标识',
    bookId    integer not null,
    constraint `BookChapter_bookId_fkey` foreign key references `Book` (`id`)
        chapterName varchar (50) not null unique comment '章节名称',
    state     char(1)            default 0 comment '章节状态',
    content   text               default '' comment '章节内容',
    readCount integer            default 0 comment '阅读量'
)
-- 订阅书籍，通过state：【0：互不关注，1：用户1关注用户2，2：用户2关注用户1，3：用户1和2互相关注】
create table `AuthorSubscribe`
(
    id      integer unique   default uuid() comment '订阅标识',
    fUserId integer not null,
    constraint `BookSubscribe_fUserId_fkey` foreign key (fUserId) references `Author` (`id`)
        sUserId integer not null,
    constraint `BookSubscribe_sUserId_fkey` foreign key (sUserId) references `Author` (`id`),
    state   char(1) not null default 0
)
```

## 外键 foreign key

外键是用于关联两张表之间的关系，如教师和学生之间，是一个教师带领一群学生，这样就需要在学生表中设置外键指向教师表，之后若是需要查找某个教师带领的所有学生，或者查找某个学生与其教师即可通过这种关联查找。
前面创建表结构时演示过直接附加外键，以下通过额外修改表的方式添加外键：

```sql
alter table `Author`
    add constraint `Author_userId_fkey` foreign key (`userId`) references `User` (`id`);
```

如果只是数据库的外键，一般是指物理外键，但也可以不依靠数据库，单纯通过代码方式进行关联，这种称为逻辑外键。

* 物理外键，对某列使用foreign key定义外键关联其他表，优点是关系都在数据库中定好，不会轻易出现人为错误。缺点是影响增删改效率，容易引发死锁，也不适用于分布式、集群和微服务等场景。
* 逻辑外键，只在代码里定义关联，所有增删改查都需要经过这层处理，优点是易于改动和处理复杂情况，缺点是缺乏直接关联，一旦有漏网之鱼没有经过中间层处理或直接操作数据库则会出现错乱。

# 查询

通过语句`select * from [tableName]`查询

```sql
select *
from User;
select *
from Author;
select bookName, state
from Book;
```

## 条件查询 where

通过条件查询指定含有符合要求的字段值才能输出对应行，这可以完成大部分日常要求。
条件查询根据字段类型可以使用：>, >=, <=, !=, between...and, in, like, is null, and ,or, not

## 聚合函数

聚合函数可以按照列（字段）进行整理操作，包含：

* sum 计算总和。
* count 计算列数，忽略Null值的列
* max 获取最大值的那一列
* min 获取最小值的那一列
* avg 获取平均值

## 分组查询 group by/having

分组查询既将查询到的结果再按照要求将相同字段的合并为一行。
包含group by 字段名 和 having 筛选条件。
以下示例是将用户查询结果按照性别分组，并计算合并的行数（既性别人数）

```sql
select *, count(id)
from User
where sex!=Null
group by sex
```

## 流程控制 case

流程控制语句可以用来根据字段值输出不同的内容，如性别定义为整型或布尔型时，可以通过这种方式输出更具有意义的文字male和female：

```sql
select (case sex when 1 then 'male' when 2 then 'female' else 'unknown' end) sex, name
from User;
```

## 排序

主要为升序asc和降序desc，默认按照升序。

```sql
select *
from User asc;
```

## 多表联查

通过在from后添加多个表名，使用where指定正确的匹配即可，如果没有使用where筛选条件将会变成笛卡尔积，既获得左表列数n和右表列数m的乘积数量的行数，左表5条记录，右表6条记录，结果是30条记录:

```sql
select User.*, Author.*
from User,
     Author
where User.id = Author.userId;
```

## 连接查询 join

通过在查询语句后面加入[join 表名 on 条件]即可进行连接查询，可细分为多种，默认为left join：

* inner join，内连接，交集，表之间能被where匹配到的行记录才会显示。在from后添加多个表时实际是使用inner join进行查询
* left join，左外连接，左全集，右交集，左表全部行记录会显示，右表被左表where匹配到的行记录才会显示
* right join，右外连接，右全集，左交集，和上述相反。
* full join，全连接，左右全集

```sql
select User.*, Author.*
from `User`
         join `Author`
where `User`.`authorId` = `Author`.`id`
``` 

三表联查：

```sql
select User.*, Author.*, Book.*
from (User join Author where User.authorId=Author.id)
         join Book
where Author.id = Book.authorId
```

## 子查询

通过在from和where中添加已经查询好的结果进行二次查询，其中条件部分称为子查询。
子查询可以返回多种不同形式的结果：

### 标量子查询

子查询中只返回一列的单个值，用于=，<=等的匹配方式

```sql
select *
from `User`
where `id` = (select userId as id from `Author` where nickname = '很爱很爱你')
```

### 列子查询

子查询中返回多列，用于in，not in匹配方式

```sql
select *
from `User`
where `id` in (select userId as id from `Author` where state = '1' or state = '2')
```

### 行子查询

子查询中返回一行（多列），使用列进行匹配，类似连接查询
```sql
select * from `Author` where (category,level)=(select category,level from `Author` where nickname='尾鱼');
```

### 表子查询
子查询中返回多行多列，类似双表查询，方便用做临时表时使用，主要用于in匹配方式
```sql
select * from (select * from `User` where createTime > '2023-01-01') u, `Author` where u.id=Author.userId;
```


# 添加/插入数据

通过语句`insert into [tableName] values (Array<field>)`插入数据

# 修改/更新数据

通过语句`update [tableName] set [field]=[value] where [condition]`：

```sql
update `Author`
set penname='笔下留情'
where id = 1;
```

# 删除数据

通过语句`delete from [tableName] where [condition]`：

```shell
delete from `BookChapter` where id=1;
```